"""Run auditable, nested regularizer comparisons on recovered literature inputs."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys,json,argparse,traceback
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation.literature_benchmark import *
from src.evaluation.market_ssvi import implied_volatility_from_normalized_call


def truth_albani(x,t):
    return np.where(abs(x)<=.4,.4-.16*np.exp(-t/2)*np.cos(4*np.pi*x/5),.4)


def generate_albani(output):
    Ts=np.repeat(np.arange(1,6)*.1,31);xs=np.tile(np.linspace(-.75,.75,31),5)
    e=CorrectionPDE(truth_albani,xs,Ts,np.linspace(0,.575,7),np.linspace(-1.05,1.05,15),
                    PDEConfig(x_limit=5,nx=1601,nt=1200))
    result=e.evaluate(np.zeros(e.n_parameters));prices=result['prices']
    fine=CorrectionPDE(truth_albani,xs,Ts,e.parameter_t,e.parameter_x,
                    PDEConfig(x_limit=5,nx=2401,nt=1800))
    refined=fine.evaluate(np.zeros(fine.n_parameters))['prices']
    diagnostic=dict(max_abs_difference=float(max(abs(prices-refined))),
        rmse_difference=float(np.sqrt(np.mean((prices-refined)**2))),
        price_noise_sd_mean=float(np.mean(.01*refined)),
        note='Published local-volatility truth/grid/noise; S0 known and fixed at 1; new noise seeds. Independent finer generation grid. Not the authors original random realization or spot-adjustment experiment.')
    (output/'albani_generation_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
    frames={}
    for seed in range(10):
        rng=np.random.default_rng(20261022+seed)
        scale=np.maximum(.01*refined,1e-6); observed=refined+scale*rng.standard_normal(len(refined))
        iv=[]
        for c,x,t in zip(observed,xs,Ts):
            try:iv.append(implied_volatility_from_normalized_call(float(c),float(x),float(t)))
            except ValueError:iv.append(np.nan)
        frame=normalized_quote_frame(Ts,xs,iv,price=observed,scale=scale)
        frame['clean_price']=refined
        frames[f'albani_noise_{seed:02d}']=frame
    return frames


def execute_task(task):
    dataset,protocol,fold,frame,train,test,outdir,config,methods,*extra=task
    shape=tuple(extra[0]) if extra else (7,15)
    out=Path(outdir)/dataset/protocol/f'fold_{fold:02d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return str(out),'cached'
    try:
        result=run_fold(frame,train,test,cfg=PDEConfig(**config),methods=methods,shape=shape)
        records=[];test=np.asarray(test)
        for name,item in result['outputs'].items():
            pred=item['prices'][test]
            for j,price in zip(test,pred):
                records.append(dict(dataset=dataset,protocol=protocol,fold=fold,index=int(j),
                    method=name,maturity=float(frame.maturity.iloc[j]),log_moneyness=float(frame.log_moneyness.iloc[j]),
                    observed_price=float(frame.price.iloc[j]),scale=float(frame.scale.iloc[j]),
                    true_iv=float(frame.iv.iloc[j]),predicted_price=float(price)))
            np.savez_compressed(out/f'{name}_calibration.npz',h=item['h'],prices=item['prices'],
                parameter_t=result['engine'].parameter_t,parameter_x=result['engine'].parameter_x)
        if dataset.startswith('albani'):
            evT=np.linspace(.1,.5,25);evx=np.linspace(-.75,.75,121)
            # Evaluate effective correction with the exact shared callback.
            from src.inverse.reference_correction import make_reference_plus_correction_local_volatility
            prior=make_market_ssvi_local_volatility_function(result['fit'])
            truth=np.array([truth_albani(evx,t) for t in evT])
            recovery=[]
            for name,item in result['outputs'].items():
                fn=make_reference_plus_correction_local_volatility(prior,item['h'].reshape(result['engine'].shape),
                    result['engine'].parameter_t,result['engine'].parameter_x,
                    maturity_taper=min(.08,.08*np.ptp(result['engine'].parameter_t)),log_moneyness_taper=.1)
                surface=np.array([fn(evx,t) for t in evT]);error=surface-truth
                for region,mask in [('full',np.ones_like(truth,dtype=bool)),('central',np.broadcast_to(abs(evx)[None,:]<=.35,truth.shape))]:
                    recovery.append(dict(method=name,region=region,rmse=float(np.sqrt(np.mean(error[mask]**2))),mae=float(np.mean(abs(error[mask])))))
                np.savez_compressed(out/f'{name}_recovery.npz',sigma=surface,truth=truth,T=evT,x=evx)
            pd.DataFrame(recovery).to_csv(out/'recovery.csv',index=False)
        pd.DataFrame(records).to_csv(out/'predictions.csv',index=False)
        pd.DataFrame(result['curves']).to_csv(out/'validation_curves.csv',index=False)
        fit=result['fit']; fitmeta={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in fit.__dict__.items()}
        summary=dict(dataset=dataset,protocol=protocol,fold=fold,config=config,parameter_shape=list(shape),
            train_indices=result['train_indices'],test_indices=result['test_indices'],ssvi_fit=fitmeta,
            selections={k:('reference' if np.isinf(v) else v) for k,v in result['selections'].items()},
            solver={k:{key:v[key] for key in ['stop','seconds','bound_fraction'] if key in v} for k,v in result['outputs'].items()},
            histories={k:v['history'] for k,v in result['outputs'].items()})
        (out/'complete.json').write_text(json.dumps(summary,indent=2))
        return str(out),'complete'
    except Exception:
        (out/'error.txt').write_text(traceback.format_exc())
        return str(out),'ERROR '+traceback.format_exc().splitlines()[-1]


def summarize(output):
    files=list(output.glob('*/*/fold_*/predictions.csv'))
    if not files:return
    predictions=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    predictions['error']=predictions.predicted_price-predictions.observed_price
    predictions['weighted_sq']=(predictions.error/predictions.scale)**2
    predictions['predicted_iv']=np.nan
    for idx,row in predictions.iterrows():
        try:
            predictions.loc[idx,'predicted_iv']=implied_volatility_from_normalized_call(row.predicted_price,row.log_moneyness,row.maturity)
        except (ValueError,RuntimeError):pass
    rows=[]
    for keys,g in predictions.groupby(['dataset','protocol','method']):
        mask=np.isfinite(g.true_iv)&np.isfinite(g.predicted_iv)
        rows.append(dict(dataset=keys[0],protocol=keys[1],method=keys[2],n=len(g),
            price_rmse=float(np.sqrt(np.mean(g.error**2))),price_mae=float(np.mean(abs(g.error))),
            scaled_price_rmse=float(np.sqrt(np.mean(g.weighted_sq))),
            iv_rmse=float(np.sqrt(np.mean((g.predicted_iv[mask]-g.true_iv[mask])**2))) if mask.any() else None,
            valid_iv_fraction=float(mask.mean())))
    predictions.to_csv(output/'all_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'pooled_metrics.csv',index=False)
    recovery=[]
    for p in output.glob('albani*/*/fold_*/recovery.csv'):
        d=pd.read_csv(p);d['dataset']=p.parts[-4];recovery.append(d)
    if recovery:
        pd.concat(recovery,ignore_index=True).to_csv(output/'all_recovery.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/stage_23_literature_audit_final')
    parser.add_argument('--mode',choices=['market','synthetic','all'],default='all')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    datasets=load_literature_inputs(args.inputs); tasks=[];cfg=asdict(PDEConfig(nx=401,nt=240,x_limit=3.))
    if args.mode in ['market','all']:
        for name,frame in datasets.items():
            frame.to_csv(output/f'{name}_prepared.csv',index=False)
            labels=interleaved_labels(frame,5)
            for fold in range(5):
                tasks.append((name,'quote_cv',fold,frame,np.where(labels!=fold)[0],np.where(labels==fold)[0],str(output),cfg,METHODS))
            for fold,t in enumerate(sorted(frame.maturity.unique())):
                test=np.where(frame.maturity.to_numpy()==t)[0];train=np.where(frame.maturity.to_numpy()!=t)[0]
                tasks.append((name,'expiry_cv',fold,frame,train,test,str(output),cfg,METHODS))
    if args.mode in ['synthetic','all']:
        for name,frame in generate_albani(output).items():
            frame.to_csv(output/f'{name}_prepared.csv',index=False)
            labels=interleaved_labels(frame,5)
            tasks.append((name,'noise_recovery',0,frame,np.where(labels!=0)[0],np.where(labels==0)[0],str(output),cfg,METHODS))
    (output/'run_design.json').write_text(json.dumps(dict(config=cfg,methods=METHODS,lambdas=LAMBDAS,
        tasks=len(tasks),parameter_shape=[7,15],inner_split='4-way interleaved, fold 1 validation',
        objective='mean scaled price squared error + lambda h^T R h / p; trace-normalized R',
        correction_bound=2.5,scope='controlled penalty ablation, not exact original-paper implementation'),indent=2))
    errors=0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(execute_task,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            name,status=f.result();print(f'{i}/{len(tasks)} {name} {status}',flush=True)
            errors+=status.startswith('ERROR')
    summarize(output)
    print(f'FINISHED errors={errors}',flush=True)

if __name__=='__main__':main()
