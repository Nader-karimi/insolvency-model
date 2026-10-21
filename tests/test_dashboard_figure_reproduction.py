from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scenario_engine import Scenario,stress
from research.reproduce_dashboard_default import heuristic_score,generate

def test_default_values():
    r=stress(Scenario())
    assert abs(r['E2']+12061.25)<1e-8
    assert abs(r['lambda_final']-2.240625)<1e-10
    assert abs(r['d_final']-.3109375)<1e-10
    assert abs(r['feedback']-.1609375)<1e-10
    assert heuristic_score(1,0.05) < 1

def test_figure_and_manifest(tmp_path):
    image=tmp_path/'scenario.png'; data=tmp_path/'scenario.json'
    generate(image,data)
    assert image.stat().st_size>15_000
    obj=json.loads(data.read_text())
    assert obj['shock']==-0.25 and obj['equity']==8000.0
    assert obj['score_coefficients']['score_threshold']==1.0
