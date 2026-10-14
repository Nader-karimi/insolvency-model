"""Transparent balance-sheet stress engine used by the web dashboard.

The engine deliberately uses accounting stocks that are available in the
published separate financial statements.  Its outputs are screening proxies,
not official Basel LCR/NSFR, NPL, PD, or capital-adequacy measures.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True)
class Scenario:
    """Stress assumptions expressed as fractions between zero and one."""

    credit_loss_rate: float
    liquid_asset_haircut: float
    deposit_runoff_rate: float

    def validate(self) -> None:
        for name, value in (
            ("credit_loss_rate", self.credit_loss_rate),
            ("liquid_asset_haircut", self.liquid_asset_haircut),
            ("deposit_runoff_rate", self.deposit_runoff_rate),
        ):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


def _number(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def evaluate(row: Mapping[str, Any], scenario: Scenario) -> dict[str, Any]:
    """Apply a joint credit, haircut, and funding-runoff stress.

    Deposit runoff reduces available liquidity but is not booked as an equity
    loss.  This avoids the common double-counting error in simple dashboards.
    """

    scenario.validate()
    loans = _number(row, "gross_loans_proxy")
    liquid_assets = _number(row, "cash_and_bank_assets")
    customer_funding = _number(row, "customer_funding")
    equity = _number(row, "total_equity")
    assets = _number(row, "total_assets")

    if assets <= 0 or customer_funding <= 0 or liquid_assets < 0 or loans < 0:
        raise ValueError("balance-sheet stocks are outside the admissible range")

    credit_loss = loans * scenario.credit_loss_rate
    haircut_loss = liquid_assets * scenario.liquid_asset_haircut
    runoff = customer_funding * scenario.deposit_runoff_rate
    available_liquidity = liquid_assets * (1 - scenario.liquid_asset_haircut)
    stressed_equity = equity - credit_loss - haircut_loss
    liquidity_surplus = available_liquidity - runoff
    coverage = available_liquidity / runoff if runoff > 0 else None
    solvency_pass = stressed_equity > 0
    liquidity_pass = liquidity_surplus >= 0

    return {
        "credit_loss_rate": scenario.credit_loss_rate,
        "liquid_asset_haircut": scenario.liquid_asset_haircut,
        "deposit_runoff_rate": scenario.deposit_runoff_rate,
        "credit_loss_million_rial": credit_loss,
        "liquid_asset_haircut_loss_million_rial": haircut_loss,
        "deposit_runoff_million_rial": runoff,
        "available_liquidity_million_rial": available_liquidity,
        "liquidity_surplus_million_rial": liquidity_surplus,
        "stressed_equity_million_rial": stressed_equity,
        "post_stress_equity_to_assets": stressed_equity / assets,
        "liquidity_runoff_coverage_proxy": coverage,
        "solvency_pass": solvency_pass,
        "liquidity_pass": liquidity_pass,
        "scenario_pass": solvency_pass and liquidity_pass,
    }
