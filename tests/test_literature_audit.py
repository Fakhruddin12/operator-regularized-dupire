import numpy as np
from scipy.interpolate import PchipInterpolator
from src.evaluation.ssvi_term_structure import theta_and_derivative
from src.pricing.calibration_pde import CorrectionPDE, PDEConfig
from src.pricing.forward_normalized_pde import normalized_black_call_price
from src.regularization.scaling import build_nondimensional_regularization_matrix


def engine(nx=201, nt=100):
    return CorrectionPDE(lambda x,t: np.full_like(x, .2),
        np.array([-.2,0.,.2,0.]), np.array([.1,.1,.7,1.]),
        np.linspace(0,1.2,4), np.linspace(-.6,.6,7),
        PDEConfig(x_limit=2,nx=nx,nt=nt))


def test_original_pchip_extrapolation_counterexample_and_shared_fix():
    t=np.array([.1,.3,.6,1.]); th=np.array([.02,.04,.055,.057])
    old=PchipInterpolator(np.r_[0,t],np.r_[0,th],extrapolate=True)
    assert old.derivative()(2.) < 0
    v,d=theta_and_derivative(t,th,np.array([.001,.2,1.,2.,10.]))
    assert np.all(v>0) and np.all(d>0)
    np.testing.assert_allclose(theta_and_derivative(t,th,t)[0],th)


def test_quadrature_constant_and_affine_integrals_are_grid_invariant():
    for nT,nx in [(3,5),(9,17)]:
        T=np.linspace(0,2,nT); x=np.linspace(-1,1,nx)
        R,_,_,_=build_nondimensional_regularization_matrix(T,x,alpha_x=0,alpha_T=0,beta=1)
        z=np.ones(nT*nx)
        np.testing.assert_allclose(z@R@z,1.,atol=1e-13)
        R,_,_,_=build_nondimensional_regularization_matrix(T,x,alpha_x=1,alpha_T=1,beta=0)
        xx,tt=np.meshgrid((x+1)/2,T/2); z=(xx+2*tt).ravel()
        np.testing.assert_allclose(z@R@z,5.,atol=1e-12)


def test_discrete_tangent_matches_centered_finite_differences():
    e=engine(); h=np.random.default_rng(7).normal(0,.1,e.n_parameters)
    exact=e.evaluate(h,jacobian=True)['jacobian']
    for j in [0,4,10,17,27]:
        delta=np.zeros_like(h); delta[j]=1e-5
        numeric=(e.evaluate(h+delta)['prices']-e.evaluate(h-delta)['prices'])/2e-5
        np.testing.assert_allclose(exact[:,j],numeric,rtol=2e-5,atol=3e-10)


def test_fast_engine_uses_the_original_tapered_correction_model():
    from src.inverse.reference_correction import make_reference_plus_correction_local_volatility
    e=engine(101,60);h=np.random.default_rng(31).normal(0,.3,e.n_parameters)
    callback=make_reference_plus_correction_local_volatility(
        lambda x,t:np.full_like(x,.2),h.reshape(e.shape),e.parameter_t,e.parameter_x,
        maturity_taper=.08*np.ptp(e.parameter_t),log_moneyness_taper=.1*np.ptp(e.parameter_x))
    for i in [0,len(e.midpoints)//2,len(e.midpoints)-1]:
        fast=e.base_variance[i]*np.exp(np.sum(h[e.indices[i]]*e.basis_weights[i],axis=1))
        original=callback(e.x[1:-1],e.midpoints[i])**2
        np.testing.assert_allclose(fast,original,rtol=5e-15,atol=1e-16)


def test_pricing_convergence_and_intrinsic_and_strike_convexity():
    errors=[]
    for nx,nt in [(101,60),(401,240)]:
        e=engine(nx,nt); result=e.evaluate(np.zeros(e.n_parameters),return_surface=True)
        target=normalized_black_call_price(e.query_x,e.query_t,.2)
        errors.append(np.max(abs(result['prices']-target)))
        c=result['surface']; k=np.exp(result['x'])
        assert np.min(c-np.maximum(1-k,0)) > -1e-12
        assert np.max(np.diff(c,axis=1)) < 1e-12
        assert np.min(np.diff(np.diff(c,axis=1)/np.diff(k),axis=1)) > -1e-10
    assert errors[1] < errors[0]/3 and errors[1]<3e-5


def test_every_quote_expiry_is_an_exact_time_knot():
    e=engine(); h=np.zeros(e.n_parameters)
    result=e.evaluate(h)
    assert len(result['prices'])==4
    assert all(t in e.times for t in e.query_t)


def test_outer_holdout_values_cannot_change_training_or_tuning():
    from src.evaluation.literature_benchmark import normalized_quote_frame, run_fold
    t=np.repeat([.1,.4,.8],7);x=np.tile(np.linspace(-.25,.25,7),3)
    frame=normalized_quote_frame(t,x,.22+.04*x*x)
    test=np.array([2,9,16]);train=np.setdiff1d(np.arange(len(frame)),test)
    cfg=PDEConfig(nx=101,nt=50)
    a=run_fold(frame,train,test,cfg,methods=['operator_full'],lambdas=[.1],shape=(3,5))
    changed=frame.copy();changed.loc[test,'price']+=.1
    changed.loc[test,'iv']=1.5;changed.loc[test,'scale']*=10
    b=run_fold(changed,train,test,cfg,methods=['operator_full'],lambdas=[.1],shape=(3,5))
    assert a['selections']==b['selections']
    np.testing.assert_allclose(a['fit'].theta,b['fit'].theta,rtol=0,atol=0)
    np.testing.assert_allclose(a['outputs']['operator_full']['h'],b['outputs']['operator_full']['h'],rtol=0,atol=0)
