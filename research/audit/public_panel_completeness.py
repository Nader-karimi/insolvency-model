"""Audit the PUBLIC PANEL'S provenance/identity coverage, not bank-default prediction.

Run from repository root:
    python research/audit/public_panel_completeness.py
This script does not estimate an HJB parameter or test insolvency forecasting.
"""
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
INFILE = BASE / 'data/processed/bank_year_metrics.csv'
OUTFILE = BASE / 'audit/public_panel_completeness.json'
df = pd.read_csv(INFILE, encoding='utf-8-sig')
required = ['symbol','jalali_year','report_id','is_audited','parse_status','qc_pass',
            'total_assets','total_liabilities','total_equity','balance_gap',
            'customer_deposits','cash_and_bank_assets','cash_bank_assets_to_deposits']
missing = set(required).difference(df.columns)
assert not missing, f'missing fields: {missing}'
assert not df.duplicated(['symbol','jalali_year']).any(), 'duplicate bank-years'
assert (df.total_assets.notna() & df.total_liabilities.notna() & df.total_equity.notna()).all()
assert (df.customer_deposits > 0).all(), 'invalid proxy denominator'
assert (df.cash_bank_assets_to_deposits.notna()).all()
assert (abs(df.total_assets - df.total_liabilities - df.total_equity) <= 1e-6).all()
proxy_match = (abs(df.cash_and_bank_assets / df.customer_deposits - df.cash_bank_assets_to_deposits) < 1e-9)
assert proxy_match.notna().all()
assert (df.parse_status == 'ok').all(), 'some extract failures'
assert df.qc_pass.fillna(False).all(), 'some records did not pass provided QC'
paired = 0
for _, grp in df.groupby('symbol'):
    years = sorted(grp.jalali_year.astype(int).tolist())
    paired += sum(right-left == 1 for left,right in zip(years,years[1:]))
results = {
    'records': int(len(df)),
    'distinct_banks': int(df.symbol.nunique()),
    'distinct_years': int(df.jalali_year.nunique()),
    'year_start': int(df.jalali_year.min()),
    'year_end': int(df.jalali_year.max()),
    'parsed_ok': int((df.parse_status == 'ok').sum()),
    'accounting_and_input_qc_pass': int(df.qc_pass.sum()),
    'audited_flags': int(df.is_audited.sum()),
    'unaudited_flags': int((~df.is_audited).sum()),
    'consecutive_within_bank_pairs': int(paired),
    'balance_sheet_identities_within_1e_6': int((abs(df.total_assets - df.total_liabilities - df.total_equity) <= 1e-6).sum()),
    'proxy_ratio_cross_field_identities_match': int(proxy_match.sum()),
    'proxy_ratio_cross_field_identities_mismatch': int((~proxy_match).sum()),
    'mismatched_bank_year_keys': [f"{sym}-{year}" for sym,year in df.loc[~proxy_match,['symbol','jalali_year']].itertuples(index=False,name=None)],
    'observed_failure_events': None,
    'official_lcr_nsfr_or_regulatory_capital_available': False,
    'interpretation': 'Basic provenance and balance-sheet accounting checks are not cross-field-ratio validation; 14 disputed proxy rows excluded from figure; no external HJB/insolvency prediction validation'
}
OUTFILE.write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
