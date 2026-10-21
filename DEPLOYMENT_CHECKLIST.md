# Deploy and reviewer handoff

- [ ] Upload the complete repository files, overwriting the old root `app.py` and `requirements.txt`; retain `research/` subfolders and `.npz` archive.
- [ ] Keep Render's existing domain, connected GitHub repository and main branch; confirm Streamlit start command and build logs.
- [ ] Compare the original scenario default: equity **8,000**, stressed equity **−12,061.25**, final NPL ratio **0.3109375**, final heuristic liquidity ratio **2.240625**.
- [ ] In the HJB panel (separate default inputs), initial `w=8000/4500`, `λ=12300/15000=0.82`, `d=0.05`; model PD from the included solution is approximately **0.03905**. The HJB reference model is not the same balance-sheet shock as the original default.
- [ ] Confirm all ten interactive Plotly figures render and explanations appear below each one.
- [ ] Check stress inputs change the four scenario plots, and HJB regulatory inputs change the six HJB plots.
- [ ] Test outside-domain inputs and verify that the dashboard displays **Outside domain** instead of extrapolating.
- [ ] Ensure manuscript Data and Code Availability refers only to the GitHub repository (do not reveal the private dashboard deployment URL).
- [ ] Run `python -m pytest -q tests research/tests/test_operational_model.py` and `python tests/ui_static_smoke.py` locally.
- [ ] Tag the release and record an immutable Git commit and ZIP checksum for the journal.

This package was checked without an actual Render deployment or browser automation. The separate `Streamlit` package was unavailable in the build container, so a Plotly-generating UI smoke test and source validation were used instead of claiming a live deployed-interface test.
