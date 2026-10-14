"""Operational bank-insolvency dashboard and JSON API.

All monetary inputs must use one common unit.  The Persian interface uses
million rials by convention.  The API deliberately labels its output as a
model-implied screening result rather than a legal insolvency determination.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from operational_model import (
    BankInputs,
    OperationalParameters,
    evaluate_bank,
    load_solution,
    parameters_from_bank,
    solve_survival_hjb,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_MODEL_DIR = ROOT / "model_outputs_v4"
DOWNLOADS = {
    "bank_input_template.csv": ("model_outputs_v4", "sample_bank_inputs.csv"),
    "sample_bank_assessments.json": ("model_outputs_v4", "sample_bank_assessments.json"),
    "operational_survival_grid.csv": ("model_outputs_v4", "operational_survival_grid.csv"),
    "operational_boundaries.csv": ("model_outputs_v4", "operational_boundaries.csv"),
    "operational_convergence.csv": ("model_outputs_v4", "operational_convergence.csv"),
    "operational_surrogate_validation.csv": ("model_outputs_v4", "operational_surrogate_validation.csv"),
    "operational_sensitivity.csv": ("model_outputs_v4", "operational_sensitivity.csv"),
    "operational_summary.json": ("model_outputs_v4", "operational_summary.json"),
    "bank_year_metrics.csv": ("data", "processed", "bank_year_metrics.csv"),
    "bank_year_hjb_mapping.csv": ("data", "processed", "bank_year_hjb_mapping.csv"),
    "qc_checks.csv": ("data", "processed", "qc_checks.csv"),
    "source_line_items.csv": ("data", "raw", "source_line_items.csv"),
}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def _bank_from_payload(payload: dict[str, Any]) -> BankInputs:
    names = (
        "total_assets", "equity", "minimum_required_equity", "hqla",
        "net_cash_outflow_30d", "available_stable_funding",
        "required_stable_funding", "gross_loans", "nonperforming_loans",
        "max_additional_borrowing", "max_sellable_assets",
    )
    missing = [name for name in names if name not in payload]
    if missing:
        raise ValueError("missing fields: " + ", ".join(missing))
    try:
        values = {name: float(payload[name]) for name in names}
    except (TypeError, ValueError) as exc:
        raise ValueError("all monetary fields must be valid numbers") from exc
    bank = BankInputs(**values, name=str(payload.get("name", "Input bank"))[:120])
    bank.validate()
    return bank


@lru_cache(maxsize=24)
def _institution_solution(
    stable_funding_buffer: float,
    max_borrowing: float,
    max_asset_sale: float,
    grid: tuple[int, int, int, int],
):
    """Cache the expensive solve by normalized institutional capacity."""

    params = OperationalParameters(
        stable_funding_buffer=stable_funding_buffer,
        max_borrowing=max_borrowing,
        max_asset_sale=max_asset_sale,
    )
    return solve_survival_hjb(
        params,
        n_wealth=grid[0], n_liquidity=grid[1], n_credit=grid[2], n_controls=grid[3],
    )


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        DATA_DIR=Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)),
        MODEL_DIR=Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)),
        ASSESS_GRID=(17, 21, 17, 13),
        JSON_SORT_KEYS=False,
    )
    if test_config:
        app.config.update(test_config)

    data_dir = Path(app.config["DATA_DIR"])
    model_dir = Path(app.config["MODEL_DIR"])
    metrics = _read_optional_csv(data_dir / "processed" / "bank_year_metrics.csv")
    mapping = _read_optional_csv(data_dir / "processed" / "bank_year_hjb_mapping.csv")
    checks = _read_optional_csv(data_dir / "processed" / "qc_checks.csv")
    summary = json.loads((model_dir / "operational_summary.json").read_text(encoding="utf-8"))
    standard_solution = load_solution(model_dir / "operational_survival_solution.npz")

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; img-src 'self' data:"
        )
        return response

    @app.get("/")
    def index():
        dashboard_summary = {
            "grid_points": int(summary["grid_points"]),
            "controls": int(summary["controls"]),
            "rmse": float(summary["surrogate"]["test_rmse"]),
            "r_squared": float(summary["surrogate"]["test_r_squared"]),
            "monotonicity": min(
                summary["numerical_diagnostics"]["wealth_monotonicity_share"],
                summary["numerical_diagnostics"]["liquidity_monotonicity_share"],
                summary["numerical_diagnostics"]["credit_monotonicity_share"],
            ),
            "observations": int(len(metrics)),
            "banks": int(metrics["symbol"].nunique()) if not metrics.empty else 0,
        }
        return render_template(
            "index.html",
            summary=dashboard_summary,
            public_metrics=_records(metrics),
            public_mapping=_records(mapping),
        )

    @app.get("/health")
    def health():
        diagnostics = summary["numerical_diagnostics"]
        model_ok = bool(
            diagnostics["probability_bounds_pass"]
            and min(
                diagnostics["wealth_monotonicity_share"],
                diagnostics["liquidity_monotonicity_share"],
                diagnostics["credit_monotonicity_share"],
            ) >= 0.99
        )
        status = 200 if model_ok else 503
        return jsonify({
            "status": "ok" if model_ok else "degraded",
            "model": summary["model"],
            "grid_points": int(summary["grid_points"]),
            "probability_bounds_pass": bool(diagnostics["probability_bounds_pass"]),
            "minimum_monotonicity_share": min(
                diagnostics["wealth_monotonicity_share"],
                diagnostics["liquidity_monotonicity_share"],
                diagnostics["credit_monotonicity_share"],
            ),
        }), status

    @app.post("/api/assess")
    def assess():
        try:
            payload = request.get_json(force=True, silent=False)
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            bank = _bank_from_payload(payload)
            target_pd = float(payload.get("target_pd", 0.20))
            mode = payload.get("mode", "institution_specific")
            if mode == "standard":
                solution = standard_solution
                solve_description = "precomputed validated standard surface"
            elif mode == "institution_specific":
                p = parameters_from_bank(bank)
                grid = tuple(int(x) for x in app.config["ASSESS_GRID"])
                solution = _institution_solution(
                    round(p.stable_funding_buffer, 6),
                    round(p.max_borrowing, 6),
                    round(p.max_asset_sale, 6),
                    grid,
                )
                solve_description = "institution-specific capacity surface"
            else:
                raise ValueError("mode must be 'standard' or 'institution_specific'")
            result = evaluate_bank(bank, solution, target_pd=target_pd)
            result["solve"] = {
                "mode": mode,
                "description": solve_description,
                "grid_shape": [
                    len(solution.wealth_grid), len(solution.liquidity_grid), len(solution.credit_grid),
                ],
                "control_count": int(app.config["ASSESS_GRID"][3]) if mode != "standard" else summary["controls"],
                "units": "all monetary values in one common unit; UI convention is million rials",
            }
            return jsonify(result)
        except (TypeError, ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/model-summary")
    def api_model_summary():
        return jsonify(summary)

    @app.get("/api/public-panel")
    def api_public_panel():
        return jsonify({
            "warning": (
                "Public-statement series are auditable software-test proxies; they are not "
                "official bank LCR, NSFR, NPL, regulatory capital, or model PD measures."
            ),
            "metrics": _records(metrics),
            "mapping": _records(mapping),
        })

    @app.get("/download/<name>")
    def download(name: str):
        if name not in DOWNLOADS:
            abort(404)
        parts = DOWNLOADS[name]
        if parts[0] == "data":
            path = data_dir.joinpath(*parts[1:])
        else:
            path = ROOT.joinpath(*parts)
        if not path.exists():
            abort(404)
        return send_from_directory(path.parent, path.name, as_attachment=True)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
