"""Predeclared fold-0 sensitivity to a denser correction grid, not test tuning."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import sys,json,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.literature_benchmark import *
spec=importlib.util.spec_from_file_location('stage23',ROOT/'scripts/23_literature_benchmarks.py')
stage23=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage23)

def main():
    primary=ROOT/'outputs/stage_23_literature_audit_final'
    output=ROOT/'outputs/stage_25_parameter_grid_sensitivity';output.mkdir(parents=True,exist_ok=True)
    design=dict(shape=[9,19],selection='Fold 0 fixed in advance for each dataset; same inner validation; no outcome-dependent choice.',
                datasets=['eurostoxx_2010','bates_author','albani_noise_00'],methods=['h1','h2','operator_full'])
    (output/'design.json').write_text(json.dumps(design,indent=2))
    for dataset in design['datasets']:
        frame=pd.read_csv(primary/f'{dataset}_prepared.csv');frame.to_csv(output/f'{dataset}_prepared.csv',index=False)
        labels=interleaved_labels(frame,5)
        task=(dataset,'quote_cv',0,frame,np.where(labels!=0)[0],np.where(labels==0)[0],str(output),
              PDEConfig(nx=401,nt=240).__dict__,design['methods'],[9,19])
        print(stage23.execute_task(task),flush=True)
    stage23.summarize(output)

if __name__=='__main__':main()
