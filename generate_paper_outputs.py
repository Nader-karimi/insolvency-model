"""Regenerate descriptive tables and figures used by the article."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "paper_assets"
OUT.mkdir(exist_ok=True)

METRICS = {
    "cash_bank_assets_to_deposits": "Cash and bank assets / customer funding",
    "loan_to_deposit_proxy": "Loans / customer funding",
    "equity_to_assets": "Book equity / total assets",
    "loans_to_assets": "Loans / total assets",
}


def main() -> None:
    metrics = pd.read_csv(DATA / "bank_year_metrics.csv", encoding="utf-8-sig")
    metrics["fiscal_year"] = pd.to_datetime(
        metrics["fiscal_year_gregorian"], errors="raise"
    ).dt.year
    scenarios = pd.read_csv(DATA / "stress_scenarios.csv", encoding="utf-8-sig")
    descriptive = metrics[list(METRICS)].describe().T.reset_index(names="metric")
    descriptive["definition"] = descriptive["metric"].map(METRICS)
    descriptive.to_csv(OUT / "descriptive_summary.csv", index=False, encoding="utf-8-sig")
    metrics.groupby("fiscal_year")[list(METRICS)].mean().reset_index().to_csv(
        OUT / "annual_means.csv", index=False, encoding="utf-8-sig"
    )
    metrics.groupby("symbol")[list(METRICS)].mean().reset_index().to_csv(
        OUT / "bank_means.csv", index=False, encoding="utf-8-sig"
    )

    pass_rates = scenarios.groupby("scenario")["scenario_pass"].agg(["sum", "count", "mean"])
    ordered = pass_rates.reindex(["baseline", "adverse", "severe"])
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=180)
    colors = ["#079c93", "#e49a13", "#c43d3d"]
    bars = ax.bar(["Baseline", "Adverse", "Severe"], ordered["mean"] * 100, color=colors, width=.62)
    ax.set_ylabel("Bank-year observations passing both screens (%)")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    for bar, value, count in zip(bars, ordered["mean"] * 100, ordered["sum"]):
        ax.text(bar.get_x() + bar.get_width()/2, value + 2, f"{value:.1f}% ({int(count)}/48)", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "scenario_pass_rates.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    result = {
        "observations": int(len(metrics)),
        "banks": int(metrics["symbol"].nunique()),
        "fiscal_years": [int(metrics["fiscal_year"].min()), int(metrics["fiscal_year"].max())],
        "audited": int(metrics["is_audited"].sum()),
        "unaudited": int((~metrics["is_audited"]).sum()),
        "qc_pass": int(metrics["qc_pass"].sum()),
        "scenario_pass": {name: int(row["sum"]) for name, row in ordered.iterrows()},
        "scenario_pass_rate": {name: float(row["mean"]) for name, row in ordered.iterrows()},
    }
    (OUT / "paper_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
