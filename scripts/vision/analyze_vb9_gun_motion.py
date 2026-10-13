"""Independently reconstruct gun-upper motion from measured joints and URDF.

This is analysis only, not an IK controller. TCP FK is checked against the
recorded simulation point before reporting gun-axis/connector statistics.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def analyze(path, start_s=12., end_s=55.):
    base=load('motion',ROOT/'scripts/vision/analyze_loaded_tool_motion.py').analyze(path,start_s,end_s)
    rows=[r for r in csv.DictReader(Path(path).open(newline='')) if start_s<=float(r['time_s'])<=end_s]
    val=lambda names:torch.tensor([[float(r[n]) for n in names] for r in rows],dtype=torch.float64)
    module=load('guide',ROOT/'source/b2_z1/b2_z1/rsl_rl/modules/pose_reference_teacher.py')
    fk=module.PoseReferenceTeacher(torch.nn.Linear(70,18),ROOT/'source/b2_z1/b2_z1/assets/b2_z1/b2_z1_gun.urdf',
                                  [8,13,14,15,16,17],[0.,2.55,-1.6,-1.4,0.,0.]).double()
    q=val([f'joint{i}' for i in range(1,7)])
    p_b,axis_b,_=fk.fk(q)
    base_pos=val(['base_'+k for k in 'xyz'])
    base_q=val(['base_q'+k for k in 'wxyz'])
    actual=val(['actual_'+k for k in 'xyz'])
    def rotate(v):
        t=2*torch.cross(base_q[:,1:],v,dim=1)
        return v+base_q[:,:1]*t+torch.cross(base_q[:,1:],t,dim=1)
    fk_error=torch.linalg.vector_norm(base_pos+rotate(p_b)-actual,dim=1)
    if torch.quantile(fk_error,.95).item()>.002:
        raise ValueError('CSV/URDF TCP frames do not agree within 2mm; refusing gun reconstruction')
    axis_w=rotate(axis_b)
    # URDF dummy TCP [0,-.03,-.62], hose connector [0,-.03,.48].
    connector=actual+1.10*axis_w
    tilt=torch.rad2deg(torch.acos(axis_w[:,2].abs().clamp(0,1)))
    span=lambda a:float(torch.quantile(a,.95)-torch.quantile(a,.05))
    limits=[]
    import xml.etree.ElementTree as ET
    root=ET.parse(ROOT/'source/b2_z1/b2_z1/assets/b2_z1/b2_z1_gun.urdf').getroot()
    for i in range(1,7):
        joint=next(j for j in root.findall('joint') if j.get('name')==f'joint{i}')
        limit=joint.find('limit');lo,hi=float(limit.get('lower')),float(limit.get('upper'))
        margin=torch.minimum(q[:,i-1]-lo,hi-q[:,i-1])
        limits.append({'joint':f'joint{i}','margin_p05_rad':float(torch.quantile(margin,.05)),
                       'within_0p01rad_fraction':float((margin<=.01).double().mean())})
    base.update({'gun_connector_z_p95_minus_p05_m':span(connector[:,2]),
                 'gun_axis_tilt_mean_deg':float(tilt.mean()),
                 'gun_axis_tilt_p95_minus_p05_deg':span(tilt),
                 'fk_tcp_error_p95_m':float(torch.quantile(fk_error,.95)),
                 'joint_limit_margins':limits,
                 'limitations':'Single continuous steady window; gun pose reconstructed from measured joints, with FK/TCP consistency check. Not load certification.'})
    return base


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces',nargs='+',type=Path)
    parser.add_argument('--start-s',type=float,default=12.)
    parser.add_argument('--end-s',type=float,default=55.)
    args=parser.parse_args()
    print(json.dumps([analyze(p,args.start_s,args.end_s) for p in args.traces],indent=2))
