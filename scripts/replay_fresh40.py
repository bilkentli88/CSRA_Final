"""Replay frozen predictions in a disposable copy, without retraining or inference."""
from pathlib import Path
import argparse,sys,json
import pandas as pd
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True,type=Path);parser.add_argument('--expected',required=True,type=Path);args=parser.parse_args()
sys.path.insert(0,str(args.run_dir/'frozen_package/code'))
from analyze import analyze
analyze(args.run_dir)
keys=['algorithm','item_id','target','strength']
actual=pd.read_csv(args.run_dir/'analysis/per_case_predictions.csv').sort_values(keys).reset_index(drop=True)
expected=pd.read_csv(args.expected/'analysis/per_case_predictions.csv').sort_values(keys).reset_index(drop=True)
if not actual[keys].equals(expected[keys]):raise AssertionError('Fresh prediction identities differ')
columns=[c for c in actual if c.endswith(('_prediction','_squared_error'))]
delta=float(np.max(np.abs(actual[columns].to_numpy()-expected[columns].to_numpy())))
if not delta<1e-7:raise AssertionError('Fresh prediction replay error: '+str(delta))
print(json.dumps({'frozen_predictions_replayed':len(actual)*7,'max_absolute_prediction_or_error_difference':delta,'refitting':False,'model_inference':False},indent=2))
