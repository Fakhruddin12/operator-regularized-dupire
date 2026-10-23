"""Fixed-grid normalized Dupire solver and exact discrete tangent sensitivities.

The PDE is the same as forward_normalized_pde.py. Discretizing in normalized
strike k=exp(x) preserves the linear intrinsic payoff exactly. A sinh x mesh
resolves the payoff kink; time knots include every requested expiry and use
quadratic spacing near zero. Backward-Euler half steps over the first two
intervals damp the initial kink. This is a deterministic pricing engine.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded


def smoothstep(z):
    z = np.clip(z, 0., 1.)
    return z*z*(3.-2.*z)


@dataclass
class PDEConfig:
    x_limit: float = 3.0
    nx: int = 401
    nt: int = 240
    clustering: float = .15
    startup_steps: int = 2


class CorrectionPDE:
    """Full-domain reference times a compact log-variance correction.

    sigma = reference * exp(taper * bilinear(h) / 2).
    Reference values and query coordinates are fixed before calibration.
    """
    def __init__(self, reference, query_x, query_t, parameter_t, parameter_x,
                 config=None, maturity_taper=None, x_taper=None):
        self.config = cfg = config or PDEConfig()
        self.query_x = np.asarray(query_x, float)
        self.query_t = np.asarray(query_t, float)
        self.parameter_t = pt = np.asarray(parameter_t, float)
        self.parameter_x = px = np.asarray(parameter_x, float)
        self.shape = (len(pt), len(px))
        self.n_parameters = len(pt)*len(px)
        if (self.query_x.shape != self.query_t.shape or self.query_x.ndim != 1
                or np.any(~np.isfinite(self.query_x))
                or np.any(~np.isfinite(self.query_t)) or np.any(self.query_t <= 0)
                or np.any(np.diff(pt) <= 0) or np.any(np.diff(px) <= 0)):
            raise ValueError("Invalid query coordinates or correction grid.")
        if cfg.nx % 2 != 1 or cfg.nx < 11 or cfg.nt < 2:
            raise ValueError("Use an odd spatial grid and at least two time steps.")
        if np.max(np.abs(self.query_x)) >= cfg.x_limit:
            raise ValueError("Quote outside the fixed PDE domain.")
        u = np.linspace(-np.arcsinh(cfg.x_limit/cfg.clustering),
                        np.arcsinh(cfg.x_limit/cfg.clustering), cfg.nx)
        self.x = cfg.clustering*np.sinh(u)
        self.k = np.exp(self.x)
        tmax = float(max(self.query_t))
        times = np.unique(np.r_[np.linspace(0., np.sqrt(tmax), cfg.nt+1)**2,
                                self.query_t])
        # Split the first two coarse intervals into implicit-Euler half steps.
        initial = times[:min(cfg.startup_steps+1, len(times))]
        extra = (initial[:-1]+initial[1:])/2
        self.times = np.unique(np.r_[times, extra])
        self.startup_end = times[min(cfg.startup_steps, len(times)-1)]
        self.dt = np.diff(self.times)
        self.midpoints = (self.times[:-1]+self.times[1:])/2
        interior = self.x[1:-1]
        left = self.k[1:-1]-self.k[:-2]
        right = self.k[2:]-self.k[1:-1]
        self.lower_stencil = self.k[1:-1]**2/(left*(left+right))
        self.upper_stencil = self.k[1:-1]**2/(right*(left+right))
        self.base_variance = np.array([
            np.broadcast_to(np.asarray(reference(interior, t), float), interior.shape)**2
            for t in self.midpoints])
        if np.any(~np.isfinite(self.base_variance)) or np.any(self.base_variance <= 0):
            raise ValueError("Invalid full-domain reference volatility; no clipping is used.")
        t_width = maturity_taper or .08*(pt[-1]-pt[0])
        x_width = x_taper or .1*(px[-1]-px[0])
        ix = np.clip(np.searchsorted(px, interior, side='right')-1, 0, len(px)-2)
        wx = np.clip((interior-px[ix])/(px[ix+1]-px[ix]), 0., 1.)
        xweight = smoothstep((interior-px[0])/x_width)*smoothstep((px[-1]-interior)/x_width)
        self.indices, self.basis_weights = [], []
        for t in self.midpoints:
            it = int(np.clip(np.searchsorted(pt, t, side='right')-1, 0, len(pt)-2))
            wt = np.clip((t-pt[it])/(pt[it+1]-pt[it]), 0., 1.)
            taper = xweight*smoothstep((t-pt[0])/t_width)*smoothstep((pt[-1]-t)/t_width)
            self.indices.append(np.array([it*len(px)+ix, it*len(px)+ix+1,
                                          (it+1)*len(px)+ix, (it+1)*len(px)+ix+1]).T)
            self.basis_weights.append(np.array([(1-wt)*(1-wx), (1-wt)*wx,
                                                wt*(1-wx), wt*wx]).T*taper[:, None])
        self.indices = np.asarray(self.indices)
        self.basis_weights = np.asarray(self.basis_weights)
        self.quote_groups = {}
        for i,t in enumerate(self.query_t):
            j = int(np.searchsorted(self.times, t))
            self.quote_groups.setdefault(j, []).append(i)
        self.quote_interpolation_left = np.clip(np.searchsorted(self.k, np.exp(self.query_x))-1, 0, cfg.nx-2)
        qleft = self.quote_interpolation_left
        self.quote_interpolation_weight = (np.exp(self.query_x)-self.k[qleft])/(self.k[qleft+1]-self.k[qleft])

    def evaluate(self, h, jacobian=False, return_surface=False):
        h = np.asarray(h, float).reshape(-1)
        if h.size != self.n_parameters or np.any(~np.isfinite(h)):
            raise ValueError("Invalid correction parameter vector.")
        c = np.maximum(1.-self.k, 0.)
        n = len(c)-2
        sens = np.zeros((n, self.n_parameters)) if jacobian else None
        prices = np.empty(len(self.query_t))
        J = np.empty((len(prices), self.n_parameters)) if jacobian else None
        surfaces = [c.copy()] if return_surface else None
        for i,dt in enumerate(self.dt):
            theta = 1. if self.times[i+1] <= self.startup_end else .5
            inds, basis = self.indices[i], self.basis_weights[i]
            log_correction = np.sum(h[inds]*basis, axis=1)
            variance = self.base_variance[i]*np.exp(log_correction)
            lo = variance*self.lower_stencil
            up = variance*self.upper_stencil
            diag = -lo-up
            # Each row uses its own diffusion coefficient (non-divergence form).
            rhs = c[1:-1]+(1-theta)*dt*(lo*c[:-2]+diag*c[1:-1]+up*c[2:])
            rhs[0] += theta*dt*lo[0]*c[0]
            rhs[-1] += theta*dt*up[-1]*c[-1]
            ab = np.zeros((3,n))
            ab[0,1:] = -theta*dt*up[:-1]
            ab[1] = 1-theta*dt*diag
            ab[2,:-1] = -theta*dt*lo[1:]
            new = c.copy()
            new[1:-1] = solve_banded((1,1), ab, rhs, check_finite=False)
            if jacobian:
                avg = theta*new+(1-theta)*c
                local_source = dt*(lo*avg[:-2]+diag*avg[1:-1]+up*avg[2:])
                srhs = sens+(1-theta)*dt*diag[:,None]*sens
                srhs[1:] += (1-theta)*dt*lo[1:,None]*sens[:-1]
                srhs[:-1] += (1-theta)*dt*up[:-1,None]*sens[1:]
                for b in range(4):
                    srhs[np.arange(n), inds[:,b]] += local_source*basis[:,b]
                sens = solve_banded((1,1), ab, srhs, check_finite=False)
            c = new
            if return_surface:
                surfaces.append(c.copy())
            if i+1 in self.quote_groups:
                q = np.array(self.quote_groups[i+1])
                j = self.quote_interpolation_left[q]
                w = self.quote_interpolation_weight[q]
                prices[q] = (1-w)*c[j]+w*c[j+1]
                if jacobian:
                    padded = np.vstack([np.zeros((1,self.n_parameters)), sens,
                                        np.zeros((1,self.n_parameters))])
                    J[q] = (1-w[:,None])*padded[j]+w[:,None]*padded[j+1]
        if np.any(~np.isfinite(prices)):
            raise FloatingPointError("Nonfinite PDE price.")
        result = {'prices': prices, 'jacobian': J}
        if return_surface:
            result.update(surface=np.asarray(surfaces), x=self.x, times=self.times)
        return result
