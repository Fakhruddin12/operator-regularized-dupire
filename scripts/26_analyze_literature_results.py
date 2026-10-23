"""Summarize complete nested benchmark results and create scientific figures."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.literature_benchmark import *
from src.evaluation.market_ssvi import make_market_ssvi_local_volatility_function
from src.inverse.reference_correction import make_reference_plus_correction_local_volatility
STAGE=ROOT/'outputs/stage_23_literature_audit_final'
OUT=ROOT/'reports/literature_audit';OUT.mkdir(parents=True,exist_ok=True)
COLORS={'ssvi_pde':'#8b99aa','unregularized':'#cab18b','h1':'#457b9d','h2':'#26927e','h1_density':'#9282b4','h1_wing':'#ba8a58','operator_full':'#d15f49'}
LABELS={'ssvi_pde':'SSVI','unregularized':'No penalty','h1':'H1','h2':'H2','h1_density':'H1+density','h1_wing':'H1+wing','operator_full':'Full operator'}


def main():
    completed=list(STAGE.glob('*/*/fold_*/complete.json'))
    diagnostics=list(STAGE.glob('*/*/fold_*/diagnostics_complete.json'))
    assert len(completed)==50 and len(diagnostics)==50,(len(completed),len(diagnostics))
    input_checks=[]
    for path in sorted(STAGE.glob('*_prepared.csv')):
        data=pd.read_csv(path);intrinsic=np.maximum(1-np.exp(data.log_moneyness),0.)
        input_checks.append(dict(dataset=path.stem.removesuffix('_prepared'),n=len(data),
            below_intrinsic=int((data.price<intrinsic-1e-12).sum()),
            above_one=int((data.price>1+1e-12).sum()),nonfinite_iv=int((~np.isfinite(data.iv)).sum())))
    (ROOT/'outputs/audit_input_diagnostics.json').write_text(json.dumps(input_checks,indent=2))
    pred=pd.read_csv(STAGE/'final_predictions.csv');metrics=pd.read_csv(STAGE/'final_metrics.csv')
    final=metrics[(metrics.evaluation=='refined_price')&(~metrics.dataset.str.startswith('albani'))]
    final.to_csv(OUT/'market_and_bates_metrics.csv',index=False)
    for (dataset,protocol,method),g in pred[~pred.dataset.str.startswith('albani')].groupby(['dataset','protocol','method']):
        expected=155 if dataset=='eurostoxx_2010' else 324
        assert len(g)==expected and g['index'].nunique()==expected,(dataset,protocol,method,len(g))
    folds=[];pairs=[];rng=np.random.default_rng(20261022)
    for (dataset,protocol),g in pred[~pred.dataset.str.startswith('albani')].groupby(['dataset','protocol']):
        p=g.pivot(index=['index','maturity','fold'],columns='method',values='refined_price')
        truth=g.drop_duplicates('index').set_index('index').observed_price
        err=p.sub(truth,axis=0,level='index')**2
        for fold,part in err.groupby(level='fold'):
            for method in ['h1','h2','operator_full']:
                folds.append(dict(dataset=dataset,protocol=protocol,fold=int(fold),method=method,
                    price_rmse=float(np.sqrt(part[method].mean())),
                    ssvi_price_rmse=float(np.sqrt(part.ssvi_pde.mean())),n=len(part),
                    beats_ssvi=bool(part[method].mean()<part.ssvi_pde.mean())))
        groups=[part for _,part in err.groupby(level='maturity')]
        for other in ['ssvi_pde','h1','h2']:
            chosen=rng.integers(0,len(groups),(5000,len(groups)))
            a=np.array([part.operator_full.sum() for part in groups])[chosen].sum(axis=1)
            b=np.array([part[other].sum() for part in groups])[chosen].sum(axis=1)
            # Both pooled MSEs have the same sampled count, which cancels.
            ratios=100*(np.sqrt(a/b)-1)
            pairs.append(dict(dataset=dataset,protocol=protocol,comparison='operator_full vs '+other,
                rmse_change_percent=float(100*(np.sqrt(err.operator_full.mean()/err[other].mean())-1)),
                descriptive_cluster_interval_low=float(np.quantile(ratios,.025)),
                descriptive_cluster_interval_high=float(np.quantile(ratios,.975)),clusters=len(groups)))
    pd.DataFrame(folds).to_csv(OUT/'fold_metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)
    recovery=[]
    for p in STAGE.glob('albani*/*/fold_*/final_recovery.csv'):
        d=pd.read_csv(p);d['dataset']=p.parts[-4];recovery.append(d)
    assert len(recovery)==10
    recovery=pd.concat(recovery,ignore_index=True);recovery.to_csv(OUT/'synthetic_recovery_by_seed.csv',index=False)
    summary=recovery.groupby(['method','region']).rmse.agg(['mean','median','std','min','max']).reset_index()
    summary.to_csv(OUT/'synthetic_recovery_summary.csv',index=False)
    synth_pairs=[]
    for region,g in recovery.groupby('region'):
        p=g.pivot(index='dataset',columns='method',values='rmse')
        for other in ['ssvi_pde','h1','h2']:
            synth_pairs.append(dict(region=region,comparison='operator_full vs '+other,
                wins=int((p.operator_full<p[other]).sum()),draws=len(p),
                mean_rmse_change_percent=float(100*(p.operator_full.mean()/p[other].mean()-1))))
    pd.DataFrame(synth_pairs).to_csv(OUT/'synthetic_paired_comparisons.csv',index=False)
    checks=pd.read_csv(STAGE/'numerical_checks.csv')
    opt=[json.loads(p.read_text()).get('training_only_optimization_polish',{}) for p in diagnostics]
    flags=[];lambda_checks=[]
    for p in completed:
        s=json.loads(p.read_text())
        curves=pd.read_csv(p.parent/'validation_curves.csv')
        for method,selected_lambda in s['selections'].items():
            candidates=curves[curves.method==method].lam
            at_boundary=(isinstance(selected_lambda,(int,float)) and
                         selected_lambda in [float(candidates.min()),float(candidates.max())])
            lambda_checks.append(dict(dataset=s['dataset'],protocol=s['protocol'],fold=s['fold'],
                method=method,selected_lambda=selected_lambda,minimum_tested=float(candidates.min()),
                maximum_tested=float(candidates.max()),selected_search_boundary=bool(at_boundary)))
        for method,d in s['solver'].items():
            flags.append(dict(dataset=s['dataset'],protocol=s['protocol'],fold=s['fold'],method=method,**d))
    pd.DataFrame(flags).to_csv(OUT/'optimizer_stops.csv',index=False)
    pd.DataFrame(lambda_checks).to_csv(OUT/'lambda_search_diagnostics.csv',index=False)
    selected=[d for d in flags if d['method'] in METHODS]
    numerical=dict(completed_folds=len(completed),diagnosed_folds=len(diagnostics),
        convexity_violations=int(checks.convexity_violations.sum()),calendar_violations=int(checks.calendar_violations.sum()),
        minimum_time_value=float(checks.minimum_time_value.min()),
        max_refinement_rmse=float(checks.refinement_test_rmse.max()),
        max_domain_rmse=float(checks.domain_test_rmse.max()),
        selected_correction_fits=len(selected),
        selected_fits_at_iteration_budget=sum(d['stop']=='maximum_iterations' for d in selected),
        selected_fits_with_active_bounds=sum(d.get('bound_fraction',0)>0 for d in selected),
        max_selected_bound_fraction=max(d.get('bound_fraction',0) for d in selected),
        selected_lambdas_at_search_boundary=sum(d['selected_search_boundary'] for d in lambda_checks),
        training_only_polished_fits=sum(len(d) for d in opt),
        polished_optimizer_success=sum(int(v['success']) for d in opt for v in d.values()))
    (OUT/'numerical_summary.json').write_text(json.dumps(numerical,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    methods=['ssvi_pde','unregularized','h1','h2','operator_full']
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(dataset,protocol) in zip(axes.flat,[('eurostoxx_2010','quote_cv'),('eurostoxx_2010','expiry_cv'),('bates_author','quote_cv'),('bates_author','expiry_cv')]):
        d=final[(final.dataset==dataset)&(final.protocol==protocol)].set_index('method').loc[methods]
        ax.bar([LABELS[m] for m in methods],d.price_rmse*10000,color=[COLORS[m] for m in methods])
        ax.set_title(('Euro Stoxx 50' if dataset.startswith('euro') else 'Bates synthetic')+' | '+('Quote holdout' if protocol=='quote_cv' else 'Expiry holdout'))
        ax.set_ylabel('Price RMSE (bp of spot)');ax.tick_params(axis='x',rotation=25)
    fig.savefig(OUT/'pricing_comparison.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for ax,region in zip(axes,['full','central']):
        p=recovery[recovery.region==region].pivot(index='dataset',columns='method',values='rmse')
        for j,m in enumerate(['ssvi_pde','h1','h2','operator_full']):
            vals=p[m].to_numpy();ax.scatter(j+np.linspace(-.12,.12,len(vals)),vals,color=COLORS[m],alpha=.8)
            ax.plot([j-.17,j+.17],[vals.mean(),vals.mean()],color='black',lw=2)
        ax.set_xticks(range(4),['SSVI','H1','H2','Full operator']);ax.set_title('Albani truth | '+region+' region')
        ax.set_ylabel('Local-volatility RMSE')
    fig.savefig(OUT/'synthetic_recovery.png',dpi=190);plt.close(fig)
    out=STAGE/'albani_noise_00/noise_recovery/fold_00'
    truth=np.load(out/'ssvi_pde_final_recovery.npz')
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    for ax,i in zip(axes,[0,12,24]):
        ax.plot(truth['x'],truth['truth'][i],color='black',label='Truth',lw=2)
        for m in ['ssvi_pde','h1','h2','operator_full']:
            d=np.load(out/f'{m}_final_recovery.npz');ax.plot(d['x'],d['sigma'][i],color=COLORS[m],label=LABELS[m])
        ax.set_title('T = '+str(round(float(truth['T'][i]),2)));ax.set_xlabel('Log-moneyness');ax.set_ylabel('Local volatility')
    axes[-1].legend(fontsize=8);fig.savefig(OUT/'albani_surface_slices.png',dpi=190);plt.close(fig)
    # Outcomes are not used to choose this seed or these time slices.
    print(final[['dataset','protocol','method','price_rmse','scaled_price_rmse']].to_string(index=False))
    print(summary.to_string(index=False));print(json.dumps(numerical,indent=2))

if __name__=='__main__':main()
