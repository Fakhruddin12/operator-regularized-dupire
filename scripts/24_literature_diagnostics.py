"""Reprice fixed calibrated models on refined grids; diagnose arbitrage and fits."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import sys,json,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.literature_benchmark import *
from src.evaluation.market_ssvi import implied_volatility_from_normalized_call
from scipy.optimize import minimize


def polish_training_objective(engine,frame,train,R,lam,h):
    """Training-only fallback for a final fit that reached its GN budget.

    Lambda stays frozen; neither outer prices nor outer residuals are consulted.
    """
    y=frame.price.to_numpy()[train];scale=frame.scale.to_numpy()[train]
    n=len(train);p=len(h);history=[]
    def fun(vector):
        r=engine.evaluate(vector,jacobian=True)
        residual=(r['prices'][train]-y)/scale
        J=r['jacobian'][train]/scale[:,None]
        value=float(residual@residual/n+lam*(vector@R@vector)/p)
        gradient=2*(J.T@residual/n+lam*(R@vector)/p)
        history.append(value)
        return value,gradient
    old=fun(h)[0]
    result=minimize(fun,h,method='L-BFGS-B',jac=True,bounds=[(-2.5,2.5)]*p,
        options=dict(maxiter=200,maxfun=250,ftol=1e-11,gtol=1e-6,maxls=30))
    best=result.x if result.fun<old else h
    return best,dict(success=bool(result.success),message=str(result.message),
        initial_objective=old,final_objective=float(min(old,result.fun)),
        evaluations=int(result.nfev),history=history)


def diagnostics(path):
    out=Path(path);stage=out.parents[2]
    if (out/'diagnostics_complete.json').exists():return str(out),'cached'
    s=json.loads((out/'complete.json').read_text());frame=pd.read_csv(stage/(s['dataset']+'_prepared.csv'))
    fm=s['ssvi_fit'];fm['maturities']=np.array(fm['maturities']);fm['theta']=np.array(fm['theta'])
    shape=tuple(s.get('parameter_shape',[7,15]))
    fit=SSVIFit(**fm);cfg=PDEConfig(**s['config']);engine=make_engine(fit,frame,cfg,shape)
    test=np.array(s['test_indices']); train=np.array(s['train_indices'])
    # Add a no-penalty correction to distinguish flexibility from regularization.
    unreg=fit_correction(engine,frame,train,np.eye(engine.n_parameters),0.,maximum_iterations=80)
    np.savez_compressed(out/'unregularized_calibration.npz',h=unreg['h'],prices=unreg['prices'],
        parameter_t=engine.parameter_t,parameter_x=engine.parameter_x)
    predictions=pd.read_csv(out/'predictions.csv')
    if not (predictions.method=='unregularized').any():
        rows=[]
        for j in test:
            rows.append(dict(dataset=s['dataset'],protocol=s['protocol'],fold=s['fold'],index=int(j),
                method='unregularized',maturity=frame.maturity.iloc[j],log_moneyness=frame.log_moneyness.iloc[j],
                observed_price=frame.price.iloc[j],scale=frame.scale.iloc[j],true_iv=frame.iv.iloc[j],
                predicted_price=unreg['prices'][j]))
        predictions=pd.concat([predictions,pd.DataFrame(rows)],ignore_index=True)
        predictions.to_csv(out/'predictions.csv',index=False)
    finecfg=PDEConfig(nx=801,nt=480,x_limit=3.,clustering=cfg.clustering)
    fine=make_engine(fit,frame,finecfg,shape)
    wide=make_engine(fit,frame,PDEConfig(nx=801,nt=480,x_limit=4.,clustering=cfg.clustering),shape)
    checks=[];refined=[];wide_predictions=[];polishing={};recovery=[]
    matrices=make_penalties(engine,frame.iloc[train])
    for name in predictions.method.unique():
        if name=='ssvi_direct':
            pred=ssvi_prices(fit,frame)
            for j in test:
                refined.append(dict(method=name,index=int(j),refined_price=pred[j],polished_price=pred[j]))
                wide_predictions.append(dict(method=name,index=int(j),wide_price=pred[j]))
            continue
        h=np.load(out/f'{name}_calibration.npz')['h']
        if name in s['solver'] and s['solver'][name]['stop']=='maximum_iterations':
            lam=s['selections'][name]
            if isinstance(lam,(int,float)):
                h,polishing[name]=polish_training_objective(engine,frame,train,matrices[name],lam,h)
                np.savez_compressed(out/f'{name}_polished_calibration.npz',h=h,
                    parameter_t=engine.parameter_t,parameter_x=engine.parameter_x)
        coarse=engine.evaluate(h,return_surface=True);f=fine.evaluate(h)['prices'];w=wide.evaluate(h)['prices']
        if s['dataset'].startswith('albani'):
            from src.inverse.reference_correction import make_reference_plus_correction_local_volatility
            truth_file=np.load(out/'ssvi_pde_recovery.npz')
            evT,evx,truth=truth_file['T'],truth_file['x'],truth_file['truth']
            fn=make_reference_plus_correction_local_volatility(make_market_ssvi_local_volatility_function(fit),
                h.reshape(engine.shape),engine.parameter_t,engine.parameter_x,
                maturity_taper=min(.08,.08*np.ptp(engine.parameter_t)),log_moneyness_taper=.1)
            surface=np.array([fn(evx,t) for t in evT]);err=surface-truth
            for region,mask in [('full',np.ones_like(truth,dtype=bool)),('central',np.broadcast_to(abs(evx)[None,:]<=.35,truth.shape))]:
                recovery.append(dict(method=name,region=region,rmse=float(np.sqrt(np.mean(err[mask]**2))),
                    mae=float(np.mean(abs(err[mask])))))
            np.savez_compressed(out/f'{name}_final_recovery.npz',sigma=surface,truth=truth,T=evT,x=evx)
        c=coarse['surface'];k=np.exp(coarse['x']);slopes=np.diff(c,axis=1)/np.diff(k)
        intrinsic=np.maximum(1-k,0.)
        checks.append(dict(method=name,minimum_time_value=float(np.min(c-intrinsic)),
            maximum_price_above_one=float(np.max(c-1)),maximum_positive_slope=float(slopes.max()),
            minimum_slope=float(slopes.min()),minimum_slope_increment=float(np.diff(slopes,axis=1).min()),
            minimum_calendar_increment=float(np.diff(c,axis=0).min()),
            convexity_violations=int(np.sum(np.diff(slopes,axis=1)<-1e-7)),
            calendar_violations=int(np.sum(np.diff(c,axis=0)<-1e-9)),
            refinement_test_rmse=float(np.sqrt(np.mean((f[test]-coarse['prices'][test])**2))),
            domain_test_rmse=float(np.sqrt(np.mean((w[test]-f[test])**2))),
            unregularized_stop=unreg['stop'] if name=='unregularized' else ''))
        for j in test:
            refined.append(dict(method=name,index=int(j),refined_price=f[j],polished_price=coarse['prices'][j]))
            wide_predictions.append(dict(method=name,index=int(j),wide_price=w[j]))
    pd.DataFrame(checks).to_csv(out/'numerical_checks.csv',index=False)
    if recovery:pd.DataFrame(recovery).to_csv(out/'final_recovery.csv',index=False)
    predictions=predictions.merge(pd.DataFrame(refined),on=['method','index'],validate='one_to_one')
    predictions=predictions.merge(pd.DataFrame(wide_predictions),on=['method','index'],validate='one_to_one')
    predictions.to_csv(out/'refined_predictions.csv',index=False)
    (out/'diagnostics_complete.json').write_text(json.dumps(dict(unregularized_stop=unreg['stop'],
        unregularized_bound_fraction=unreg['bound_fraction'],unregularized_history=unreg['history'],
        fine_config=finecfg.__dict__,training_only_optimization_polish=polishing),indent=2))
    return str(out),'complete'


def aggregate(stage):
    frames=[]
    for p in stage.glob('*/*/fold_*/refined_predictions.csv'):
        d=pd.read_csv(p)
        if 'polished_price' not in d:d['polished_price']=d['predicted_price']
        frames.append(d)
    if not frames:return
    predictions=pd.concat(frames,ignore_index=True);rows=[]
    for column in ['predicted_price','polished_price','refined_price','wide_price']:
        for keys,g in predictions.groupby(['dataset','protocol','method']):
            err=g[column]-g.observed_price
            iv=[]
            for _,r in g.iterrows():
                try:iv.append(implied_volatility_from_normalized_call(r[column],r.log_moneyness,r.maturity))
                except (ValueError,RuntimeError):iv.append(np.nan)
            iv=np.asarray(iv);true=g.true_iv.to_numpy();valid=np.isfinite(iv)&np.isfinite(true)
            identifiable=valid & (g.scale.to_numpy()>1.000001e-5)
            rows.append(dict(dataset=keys[0],protocol=keys[1],method=keys[2],evaluation=column,
                n=len(g),price_rmse=float(np.sqrt(np.mean(err**2))),
                scaled_price_rmse=float(np.sqrt(np.mean((err/g.scale)**2))),
                iv_rmse=float(np.sqrt(np.mean((iv[valid]-true[valid])**2))) if valid.any() else None,
                identified_iv_rmse=float(np.sqrt(np.mean((iv[identifiable]-true[identifiable])**2))) if identifiable.any() else None,
                iv_valid_fraction=float(valid.mean()),n_identified_iv=int(identifiable.sum())))
    pd.DataFrame(rows).to_csv(stage/'final_metrics.csv',index=False)
    predictions.to_csv(stage/'final_predictions.csv',index=False)
    checks=[]
    for p in stage.glob('*/*/fold_*/numerical_checks.csv'):
        d=pd.read_csv(p);d['dataset']=p.parts[-4];d['protocol']=p.parts[-3];d['fold']=p.parts[-2];checks.append(d)
    pd.concat(checks,ignore_index=True).to_csv(stage/'numerical_checks.csv',index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=3);args=parser.parse_args()
    folders=[p.parent for p in args.stage.glob('*/*/fold_*/complete.json')]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i,f in enumerate(as_completed([pool.submit(diagnostics,str(p)) for p in folders]),1):
            print(i,len(folders),f.result(),flush=True)
    aggregate(args.stage)

if __name__=='__main__':main()
