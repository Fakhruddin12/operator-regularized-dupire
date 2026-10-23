"""Leakage-controlled comparisons of penalties, using one pricing model.

These are controlled adaptations of H1/H2 regularization, not reproductions of
the original papers' parameterizations, optimizers or reported in-sample errors.
"""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.special import expit, ndtr
from scipy.sparse import diags, eye, kron
from src.evaluation.ssvi_dupire import SSVIFit, ssvi_total_variance
from src.evaluation.market_ssvi import make_market_ssvi_local_volatility_function
from src.evaluation.ssvi_term_structure import theta_and_derivative
from src.pricing.calibration_pde import CorrectionPDE, PDEConfig
from src.regularization.scaling import build_nondimensional_regularization_matrix
from src.regularization.potentials import wing_potential, quote_confidence_surface, confidence_potential

METHODS = ['h1', 'h2', 'h1_density', 'h1_wing', 'operator_full']
LAMBDAS = [1e-4, .001, .01, .1, 1., 10., 100.]


def black_from_w(x,w):
    root=np.sqrt(np.maximum(w,1e-16)); d1=-x/root+.5*root; d2=d1-root
    # Evaluate the OTM side and use parity; avoid subtracting two near-unit CDFs.
    put=np.exp(x)*ndtr(-d2)-ndtr(-d1)
    call=ndtr(d1)-np.exp(x)*ndtr(d2)
    return np.where(x<0,put+1-np.exp(x),call)


def fit_price_ssvi(train):
    """Train-only SSVI, constrained by global sufficient butterfly bounds.

    gamma in (.05,.5), eta < 2/sqrt(1+abs(rho)) conservatively ensure
    theta*phi^2*(1+abs(rho))<4 and theta*phi*(1+abs(rho))<4 for all theta>0.
    This restriction is shared by every correction method and the comparator.
    """
    start=perf_counter(); times=np.sort(train.maturity.unique()); n=len(times)
    x=train.log_moneyness.to_numpy(); y=train.price.to_numpy(); scale=train.scale.to_numpy()
    idx=np.searchsorted(times,train.maturity.to_numpy())
    dt=np.diff(np.r_[0,times])
    atm=[]
    for t in times:
        g=train[train.maturity==t].sort_values('log_moneyness')
        k=int(np.argmin(abs(g.log_moneyness.to_numpy())))
        iv=float(g.iv.iloc[k])
        atm.append(max((iv if np.isfinite(iv) and iv>0 else .25)**2*t,1e-6))
    atm=np.maximum.accumulate(atm); slopes=np.clip(np.diff(np.r_[0,atm])/dt,1e-4,2.)
    def decode(raw):
        th=np.cumsum(np.exp(raw[:n])*dt)
        rho=.99*np.tanh(raw[n]); eta=.999*2/np.sqrt(1+abs(rho))*expit(raw[n+1])
        gamma=.05+.45*expit(raw[n+2])
        return th,rho,eta,gamma
    def residual(raw):
        th,rho,eta,gamma=decode(raw)
        w=ssvi_total_variance(x,th[idx],rho,eta,gamma)
        return (black_from_w(x,w)-y)/scale
    solutions=[]
    for rho in [-.5,0.,.5]:
        raw=np.r_[np.log(slopes),np.arctanh(rho/.99),.4,0.]
        sol=least_squares(residual,raw,bounds=(np.r_[np.full(n,-13),-4,-8,-8],
                                            np.r_[np.full(n,3),4,8,8]),
                          max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-8)
        solutions.append(sol)
    sol=min(solutions,key=lambda s:s.cost); th,rho,eta,gamma=decode(sol.x)
    return SSVIFit(times,th,float(rho),float(eta),float(gamma),float(sol.cost),
                   bool(sol.success),str(sol.message),len(train),perf_counter()-start)


def ssvi_prices(fit, frame):
    th,_=theta_and_derivative(fit.maturities,fit.theta,frame.maturity.to_numpy())
    return black_from_w(frame.log_moneyness.to_numpy(),ssvi_total_variance(
        frame.log_moneyness.to_numpy(),th,fit.rho,fit.eta,fit.gamma))


def make_engine(fit, frame, cfg=None, shape=(7,15)):
    tmax=frame.maturity.max(); xmax=max(.5,float(abs(frame.log_moneyness).max())+.3)
    T=np.linspace(0,1.15*tmax,shape[0]); x=np.linspace(-xmax,xmax,shape[1])
    engine=CorrectionPDE(make_market_ssvi_local_volatility_function(fit),
        frame.log_moneyness.to_numpy(),frame.maturity.to_numpy(),T,x,cfg,
        maturity_taper=min(.08,.08*(T[-1]-T[0])),x_taper=.10)
    return engine


def make_penalties(engine, train):
    T,x=engine.parameter_t,engine.parameter_x
    conf=quote_confidence_surface(x,T,train.log_moneyness.to_numpy(),train.maturity.to_numpy(),
          log_moneyness_bandwidth=.12*(x[-1]-x[0]),maturity_bandwidth=.12*(T[-1]-T[0]))
    density=confidence_potential(conf,strength=.2)
    wing=wing_potential(x,T,strength=.5,power=2,start=.44*max(abs(x)))
    matrices={}
    for name,pot in [('h1',None),('h1_density',density),('h1_wing',wing),('operator_full',density+wing)]:
        R,_,_,_=build_nondimensional_regularization_matrix(T,x,.005,.002,1e-4,pot)
        matrices[name]=R.toarray()
    # Second derivative penalties in each dimension; a conventional comparator
    # in the same correction coordinates, not Geng's exact volatility-space fit.
    def d2(n):
        return diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)).tocsc()*(n-1)**2
    Dx=kron(eye(len(T)),d2(len(x))); Dt=kron(d2(len(T)),eye(len(x)))
    R=1e-4*eye(engine.n_parameters)+.005*(Dx.T@Dx)+.002*(Dt.T@Dt)
    matrices['h2']=R.toarray()
    # Remove arbitrary global scale differences; each gets its own tuned lambda.
    for name,R in matrices.items():
        matrices[name]=R/(np.trace(R)/len(R))
    return matrices


def fit_correction(engine, frame, fit_indices, R, lam, maximum_iterations=50):
    indices=np.asarray(fit_indices,int); y=frame.price.to_numpy()[indices]
    scale=frame.scale.to_numpy()[indices]; n=len(indices); p=engine.n_parameters
    h=np.zeros(p); history=[]; stop='maximum_iterations'; started=perf_counter()
    def objective(prices, vector):
        r=(prices[indices]-y)/scale
        return float(r@r/n+lam*(vector@R@vector)/p)
    current=engine.evaluate(h,jacobian=True); obj=objective(current['prices'],h)
    for it in range(maximum_iterations):
        J=current['jacobian'][indices]/scale[:,None]
        residual=(current['prices'][indices]-y)/scale
        g=J.T@residual/n+lam*(R@h)/p
        G=J.T@J/n+lam*R/p
        pg=np.where(((h<=-2.5+1e-8)&(g>0))|((h>=2.5-1e-8)&(g<0)),0,g)
        history.append(dict(iteration=it,objective=obj,gradient_inf=float(max(abs(pg)))))
        if max(abs(pg))<1e-6:
            stop='gradient_tolerance';break
        damping=1e-7*max(1.,np.trace(G)/p); accepted=False
        for attempt in range(5):
            step=solve(G+damping*np.eye(p),-g,assume_a='pos',check_finite=False)
            largest=max(abs(step))
            if largest>.7: step*=.7/largest
            for alpha in [1.,.5,.25,.125,.0625,.03125,.015625]:
                candidate=np.clip(h+alpha*step,-2.5,2.5)
                prediction=engine.evaluate(candidate)
                value=objective(prediction['prices'],candidate)
                if value<obj-1e-12:
                    old=obj;obj=value;h=candidate;accepted=True;break
            if accepted:break
            damping*=100
        if not accepted:
            stop='no_objective_decrease';break
        current=engine.evaluate(h,jacobian=True)
        if old-obj<1e-7*max(1.,old):
            stop='relative_objective_tolerance';break
    final=engine.evaluate(h)
    return dict(h=h,prices=final['prices'],objective=obj,stop=stop,history=history,
                seconds=perf_counter()-started,bound_fraction=float(np.mean(abs(h)>2.5-1e-7)))


def interleaved_labels(frame, nfolds=5):
    labels=np.empty(len(frame),int)
    for _,g in frame.groupby('maturity',sort=True):
        order=g.sort_values('log_moneyness').index.to_numpy()
        labels[order]=np.arange(len(order))%nfolds
    return labels


def run_fold(frame, train_idx, test_idx, cfg=None, methods=METHODS, lambdas=LAMBDAS, shape=(7,15)):
    train=frame.iloc[train_idx].reset_index(drop=True)
    inner_labels=interleaved_labels(train,4)
    inner_fit=np.where(inner_labels!=1)[0]; inner_val=np.where(inner_labels==1)[0]
    if len(inner_val)==0: raise ValueError('Empty validation split')
    inner_ssvi=fit_price_ssvi(train.iloc[inner_fit]); inner_engine=make_engine(inner_ssvi,frame,cfg,shape)
    inner_R=make_penalties(inner_engine,train.iloc[inner_fit])
    inner_full_fit=np.asarray(train_idx)[inner_fit]; inner_full_val=np.asarray(train_idx)[inner_val]
    curves=[]; selections={}
    baseline=inner_engine.evaluate(np.zeros(inner_engine.n_parameters))['prices']
    validation_score=lambda pred: float(np.mean(((pred[inner_full_val]-frame.price.to_numpy()[inner_full_val])/frame.scale.to_numpy()[inner_full_val])**2))
    base_score=validation_score(baseline)
    for method in methods:
        candidates=[(float('inf'),base_score)]
        for lam in lambdas:
            result=fit_correction(inner_engine,frame,inner_full_fit,inner_R[method],lam)
            score=validation_score(result['prices']);candidates.append((lam,score))
            curves.append(dict(method=method,lam=lam,validation_mse=score,
                               iterations=len(result['history']),stop=result['stop'],
                               bound_fraction=result['bound_fraction']))
        selected=min(candidates,key=lambda x:x[1])[0]
        # One deterministic extension when the preferred candidate is an edge.
        extra=[]
        if selected==min(lambdas):extra=[min(lambdas)/100,min(lambdas)/10]
        elif selected==max(lambdas):extra=[max(lambdas)*10,max(lambdas)*100]
        for lam in extra:
            result=fit_correction(inner_engine,frame,inner_full_fit,inner_R[method],lam)
            score=validation_score(result['prices']);candidates.append((lam,score))
            curves.append(dict(method=method,lam=lam,validation_mse=score,
                               iterations=len(result['history']),stop=result['stop'],
                               bound_fraction=result['bound_fraction']))
        selections[method]=min(candidates,key=lambda x:x[1])[0]
    fit=fit_price_ssvi(train); engine=make_engine(fit,frame,cfg,shape); R=make_penalties(engine,train)
    zero=np.zeros(engine.n_parameters); base=engine.evaluate(zero)['prices']
    outputs={'ssvi_pde':dict(h=zero,prices=base,stop='baseline',seconds=0.,history=[]),
             'ssvi_direct':dict(h=zero,prices=ssvi_prices(fit,frame),stop='baseline',seconds=0.,history=[])}
    for method in methods:
        lam=selections[method]
        if np.isinf(lam):outputs[method]=dict(h=zero,prices=base,stop='validation_selected_reference',seconds=0.,history=[],bound_fraction=0.)
        else:outputs[method]=fit_correction(engine,frame,train_idx,R[method],lam)
        outputs[method]['lam']=lam
    # Return complete train/test provenance and parameters for independent repricing.
    return dict(outputs=outputs,curves=curves,selections=selections,fit=fit,engine=engine,
                train_indices=list(map(int,train_idx)),test_indices=list(map(int,test_idx)))


def normalized_quote_frame(T,x,iv,price=None,scale=None):
    T=np.asarray(T,float);x=np.asarray(x,float);iv=np.asarray(iv,float)
    if price is None:price=black_from_w(x,iv**2*T)
    root=iv*np.sqrt(T);d1=-x/root+.5*root
    vega=np.exp(-.5*d1*d1)/np.sqrt(2*np.pi)*np.sqrt(T)
    if scale is None:scale=np.maximum(.005*vega,1e-5)
    return pd.DataFrame(dict(maturity=T,log_moneyness=x,iv=iv,price=price,scale=scale))


def load_literature_inputs(directory):
    directory=Path(directory)
    raw=json.loads((directory/'eurostoxx_2010_155_quotes.json').read_text())
    data=pd.DataFrame(raw['quotes'])
    euro=normalized_quote_frame(data.maturity_years_label,np.log(data.strike_over_spot),data.implied_volatility)
    data=pd.read_csv(directory/'ackerer_original_bates_train_data.csv')
    bates=normalized_quote_frame(data.ttm,data.logm,data.iv,price=data['put']+1-np.exp(data.logm))
    reconstructed=black_from_w(data.logm.to_numpy(),data.iv.to_numpy()**2*data.ttm.to_numpy())
    if max(abs(bates.price-reconstructed))>1e-12: raise ValueError('Bates normalization mismatch')
    return {'eurostoxx_2010':euro,'bates_author':bates}
