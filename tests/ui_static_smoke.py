"""Smoke-exercise full app without a browser; Streamlit runtime unavailable in CI container.

Uses real Plotly objects and all default values; does not verify Render deployment.
"""
import sys,runpy,types,os
from pathlib import Path

class Stub(types.ModuleType):
    charts=[]
    def __init__(self):super().__init__('streamlit');self.sidebar=self
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def columns(self,n):return [self]*n
    def tabs(self,n):return [self]*len(n)
    def expander(self,*args,**kwargs):return self
    def cache_resource(self,*args,**kwargs):return lambda fn:fn
    def number_input(self,*args,**kwargs):return kwargs['value']
    def slider(self,*args,**kwargs):return kwargs.get('value',args[3] if len(args)>3 else None)
    def select_slider(self,*args,**kwargs):return kwargs['value']
    def plotly_chart(self,fig,**kwargs):
        fig.to_json(); self.charts.append(fig)
    def stop(self):raise RuntimeError('Streamlit stop')
    def __getattr__(self,name):
        def func(*args,**kwargs):return None
        return func

stub=Stub()
sys.modules['streamlit']=stub
root=Path(__file__).parents[1]
sys.path.insert(0,str(root))
runpy.run_path(str(root/'app.py'),run_name='__main__')
assert len(stub.charts)>=10, len(stub.charts)
print(f'App UI smoke PASS: {len(stub.charts)} real Plotly figures built and serialized.')
