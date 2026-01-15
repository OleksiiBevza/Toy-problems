import numpy as np
import math
import torch
from tqdm import tqdm
import warnings



def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Cast torch tensor to numpy.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
        Numpy array corresponding to the input tensor.
    """
    return x.detach().numpy()


def numpy_to_torch(x: np.ndarray) -> torch.Tensor:
    """
    Cast numpy array to torch tensor.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
        Torch tensor corresponding to the input array.
    """
    return torch.tensor(x, dtype=torch.float32)


def torch_double_to_float(x: torch.Tensor, warn: bool = True):
    """
    Cast double precision (Float64) torch tensor to single precision (Float32).

    Parameters
    ----------
    x: torch.Tensor
        Input tensor.
    warn: bool
        If True, warn the user about the typecast.

    Returns
    -------
        Single precision (Float32) torch tensor.
    """
    if x.dtype == torch.float64 and warn:
        warnings.warn(f"Float64 data is currently unsupported, casting to Float32. Output will also have type Float32.")
        return x.float()
    elif x.dtype == torch.float32:
        return x
    else:
        raise ValueError(f"Unsupported datatype for input data: {x.dtype}")

class flow_numpy_wrapper:
    """
    Wrapper class for numpy flows.

    Parameters
    ----------
    flow : Flow object
        Flow object that implements forward and inverse
        transformations.
    
    Returns
    -------
    Flow object
    """
    def __init__(self, flow):
        self.flow = flow

    @torch.no_grad()
    def forward(self, v):
        v = numpy_to_torch(v)
        theta, logdetj = self.flow.forward(v)
        theta = torch_to_numpy(theta)
        logdetj = - torch_to_numpy(logdetj)
        return theta, logdetj

    @torch.no_grad()
    def inverse(self, theta):
        theta = numpy_to_torch(theta)
        v, logdetj = self.flow.inverse(theta)
        v = torch_to_numpy(v)
        logdetj = torch_to_numpy(logdetj)
        return v, logdetj
    



import numpy as np
import torch

#from .tools import numpy_to_torch, torch_to_numpy, flow_numpy_wrapper
#from .student import fit_mvstud
from tools import *
from student import *

@torch.no_grad()
def preconditioned_pcn(state_dict: dict,
                       function_dict: dict,
                       option_dict: dict):
    """
    Doubly Preconditioned Crank-Nicolson
    
    Parameters
    ----------
    state_dict : dict
        Dictionary of current state
    function_dict : dict
        Dictionary of functions.
    option_dict : dict
        Dictionary of options.
    
    Returns
    -------
    Results dictionary
    """
    # Likelihood call counter
    n_calls = 0

    # Clone state variables
    u = np.copy(state_dict.get('u'))
    x = np.copy(state_dict.get('x'))
    logdetj = np.copy(state_dict.get('logdetj'))
    logl = np.copy(state_dict.get('logl'))
    logp = np.copy(state_dict.get('logp'))
    beta = state_dict.get('beta')
    blobs = state_dict.get('blobs')
    if blobs is None:
        have_blobs = False
    else:
        have_blobs = True

    # Get functions
    log_like = function_dict.get('loglike')
    log_prior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')
    flow = flow_numpy_wrapper(function_dict.get('flow'))
    geometry = function_dict.get('theta_geometry')

    # Get MCMC options
    n_max = option_dict.get('n_max')
    n_steps = option_dict.get('n_steps')
    progress_bar = option_dict.get('progress_bar')
    sigma = np.minimum(option_dict.get('proposal_scale'), 0.99)

    # Get number of particles and parameters/dimensions
    n_walkers, n_dim = x.shape

    # Transform u to theta
    theta, logdetj_flow = flow.forward(u)


    mu = geometry.t_mean
    cov = geometry.t_cov
    nu = geometry.t_nu

    inv_cov = np.linalg.inv(cov)
    chol_cov = np.linalg.cholesky(cov)

    logp2_val = np.mean(logl + logp)
    cnt = 0

    i = 0
    while True:
        i += 1

        diff = theta - mu
        s = np.empty(n_walkers)
        for k in range(n_walkers):
            s[k] = 1./np.random.gamma((n_dim + nu) / 2, 2.0/(nu + np.dot(diff[k],np.dot(inv_cov,diff[k]))))

        # Propose new points in theta space
        theta_prime = np.empty((n_walkers, n_dim))
        for k in range(n_walkers):
            theta_prime[k] = mu + (1.0 - sigma ** 2.0) ** 0.5 * diff[k] + sigma * np.sqrt(s[k]) * np.dot(chol_cov, np.random.randn(n_dim))      

        # Transform to u space
        u_prime, logdetj_flow_prime = flow.inverse(theta_prime)

        # Transform to x space
        x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Apply boundary conditions
        if (scaler.periodic is not None) or (scaler.reflective is not None):
            x_prime = scaler.apply_boundary_conditions_x(x_prime)
            u_prime = scaler.forward(x_prime, check_input=False)
            x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Compute finite mask
        finite_mask_logdetj_prime = np.isfinite(logdetj_prime)
        finite_mask_x_prime = np.isfinite(x_prime).all(axis=1)
        finite_mask = finite_mask_logdetj_prime & finite_mask_x_prime

        # Evaluate prior
        logp_prime = np.empty(n_walkers)
        logp_prime[finite_mask] = log_prior(x_prime[finite_mask])
        logp_prime[~finite_mask] = -np.inf
        finite_mask_logp = np.isfinite(logp_prime)
        finite_mask = finite_mask & finite_mask_logp
        
        # Evaluate likelihood
        logl_prime = np.empty(n_walkers)
        if have_blobs:
            blobs_prime = np.empty(n_walkers, dtype=np.dtype((blobs[0].dtype, blobs[0].shape)))
            logl_prime[finite_mask], blobs_prime[finite_mask] = log_like(x_prime[finite_mask])
        else:
            logl_prime[finite_mask], _ = log_like(x_prime[finite_mask])
        logl_prime[~finite_mask] = -np.inf
        
        # Update likelihood call counter
        n_calls += np.sum(finite_mask)

        # Compute Metropolis factors
        diff_prime = theta_prime-mu
        A = np.empty(n_walkers)
        B = np.empty(n_walkers)
        for k in range(n_walkers):
            A[k] = -(n_dim+nu)/2*np.log(1+np.dot(diff_prime[k],np.dot(inv_cov,diff_prime[k]))/nu)
            B[k] = -(n_dim+nu)/2*np.log(1+np.dot(diff[k],np.dot(inv_cov,diff[k]))/nu)
        alpha = np.minimum(
            np.ones(n_walkers),
            np.exp(logl_prime * beta - logl * beta + logp_prime - logp + logdetj_prime - logdetj + logdetj_flow_prime - logdetj_flow - A + B)
        )
        alpha[np.isnan(alpha)] = 0.0

        # Metropolis criterion
        u_rand = np.random.rand(n_walkers)
        mask = u_rand < alpha

        # Accept new points
        theta[mask] = theta_prime[mask]
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        logdetj[mask] = logdetj_prime[mask]
        logdetj_flow[mask] = logdetj_flow_prime[mask]
        logl[mask] = logl_prime[mask]
        logp[mask] = logp_prime[mask]
        if have_blobs:
            blobs[mask] = blobs_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        sigma = np.abs(np.minimum(sigma + 1 / (i + 1)**0.75 * (np.mean(alpha) - 0.234), np.minimum(2.38 / n_dim**0.5, 0.99)))
        #sigma = np.minimum(sigma + 1 / (i + 1)**0.5 * (np.mean(alpha) - 0.234), 0.99)

        # Adapt mean parameter using diminishing adaptation
        mu = mu + 1.0 / (i + 1.0) * (np.mean(theta, axis=0) - mu)

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(
                dict(calls=progress_bar.info['calls'] + np.sum(finite_mask),
                    acc=np.mean(alpha),
                    steps=i,
                    logP=np.mean(logl + logp),
                    eff=sigma / (2.38 / np.sqrt(n_dim)),
                    )
            )

        # Loop termination criteria:
        logp2_val_new = np.mean(logl + logp)
        if logp2_val_new > logp2_val:
            cnt = 0
            logp2_val = logp2_val_new
        else:
            cnt += 1
            if cnt >= n_steps * ((2.38 / n_dim**0.5) / sigma)**2.0:
                break

        if i >= n_max:
            break

    return dict(u=u, x=x, logdetj=logdetj, logl=logl, logp=logp, blobs=blobs, efficiency=sigma, 
                accept=np.mean(alpha), steps=i, calls=n_calls, proposal_scale=sigma)