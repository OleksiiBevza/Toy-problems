##################################################################################
# EXPERIMENT RUNNER 
##################################################################################
import json
import matplotlib.pyplot as plt
import corner
import os
import re
import numpy as np

import jax
import jax.numpy as jnp


from pcn import *
from likelihood import *
from gaussian_mixture import *
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)



SUPPORTED_EXPERIMENTS = {"gaussian"}


# -----------------------------------------------------------------------------
# Helpers (since we do not use flow)
# -----------------------------------------------------------------------------
def mixture_mean_cov(means: jnp.ndarray, covs: jnp.ndarray, weights: jnp.ndarray, jitter: float = 1e-6):
    """
    Compute mixture mean/cov for a Gaussian mixture:
      means:   (K, D)
      covs:    (K, D, D)
      weights: (K,)
    Returns:
      mu:  (D,)
      cov: (D, D)
    """
    w = weights / jnp.sum(weights)
    mu = jnp.sum(w[:, None] * means, axis=0)  # (D,)

    diff = means - mu[None, :]  # (K, D)
    outer = diff[:, :, None] * diff[:, None, :]  # (K, D, D)

    cov = jnp.sum(w[:, None, None] * (covs + outer), axis=0)
    cov = cov + jitter * jnp.eye(cov.shape[0], dtype=cov.dtype)
    return mu, cov


def make_uniform_box_logprior(low: jnp.ndarray, high: jnp.ndarray):
    """
    Uniform prior on a hyper-rectangle:
      log p(x) = 0 inside [low, high], -inf outside
    """
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    def logprior_fn(x: jnp.ndarray) -> jnp.ndarray:
        inside = jnp.all((x >= low) & (x <= high))
        return jax.lax.select(
            inside,
            jnp.asarray(0.0, dtype=x.dtype),
            jnp.asarray(-jnp.inf, dtype=x.dtype),
        )

    return logprior_fn


class IdentityBijection:
    def transform_and_log_det(self, u, condition=None):
        return u, jnp.asarray(0.0, dtype=u.dtype)

    def inverse_and_log_det(self, theta, condition=None):
        return theta, jnp.asarray(0.0, dtype=theta.dtype)


class IdentityFlow:
    """Use this only to validate wiring when you don't have a trained FlowJAX flow yet."""
    def __init__(self):
        self.bijection = IdentityBijection()









##################################################################################
# Runner
##################################################################################
class pcn_ExperimentRunner:
    """
    Runner that can generate samples using preconditioned_pcn_jax.

    Key design decisions:
      - Kernel runs multiple times (n_outer), each time updating the full ensemble state.
      - x is stored across outer iterations: samples shape (n_outer, N_walkers, D).
    """

    def __init__(self, args, *, flow=None, scaler_cfg=None, scaler_masks=None):
        self.params = vars(args)

        # --- unique outdir ---
        base_results_dir = self.params["outdir"]
        unique_outdir = self.get_next_available_outdir(base_results_dir)
        print(f"Using output directory: {unique_outdir}")
        os.makedirs(unique_outdir, exist_ok=False)
        self.params["outdir"] = unique_outdir

        # --- validate experiment ---
        if self.params["experiment_type"] not in SUPPORTED_EXPERIMENTS:
            raise ValueError(
                f"Experiment type {self.params['experiment_type']} is not supported. "
                f"Supported types are: {SUPPORTED_EXPERIMENTS}"
            )

        print("Passed parameters:")
        for k, v in self.params.items():
            print(f"{k}: {v}")

        # --- attachment of the flow + scaler ---
        self.flow = flow
        self.scaler_cfg = scaler_cfg
        self.scaler_masks = scaler_masks

        # --- setup experiment --- 
        if self.params["experiment_type"] == "gaussian":
            self._setup_gaussian_experiment(args)

        # --- initiate variables results ---
        self.samples = None
        self.accept_history = None
        self.sigma_history = None
        self.calls_history = None

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    def _setup_gaussian_experiment(self, args):
        print("Setting the target function to a Gaussian mixture distribution.")

        np.random.seed(900)

        D = int(self.params["n_dims"])

        # generate true samples and parameters 
        true_samples, means, covariances, weights = GaussianMixtureGenerator.generate_gaussian_mixture(
            n_dim=D,
            n_gaussians=args.nr_of_components,
            n_samples=args.nr_of_samples,
            width_mean=args.width_mean,
            width_cov=args.width_cov,
            weights=args.weights_of_components,
        )

        self.true_samples = true_samples

        # convert parameterss to JAX arrays
        self.mcmc_means = jnp.stack(means, axis=0)        # (K, D)
        self.mcmc_covs = jnp.stack(covariances, axis=0)   # (K, D, D)
        self.mcmc_weights = jnp.asarray(weights)          # (K,)

        # initiate Likelihood 
        self.likelihood = GaussianMixtureLikelihood(
            means=self.mcmc_means,
            covs=self.mcmc_covs,
            weights=self.mcmc_weights,
        )

        # prior bounds: they are uniform
        low_np, high_np = self.make_auto_bounds_inflated(
            means=means,
            covs=covariances,
            inflate=float(self.params.get("prior_inflate", 9.0)),
            nsig=float(self.params.get("prior_nsig", 12.0)),
            pad=float(self.params.get("prior_pad", 1e-6)),
        )
        self.prior_low = jnp.asarray(low_np)
        self.prior_high = jnp.asarray(high_np)

        # student-t in theta-space: mixture moments by default)
        self.geom_mu, self.geom_cov = mixture_mean_cov(
            self.mcmc_means, self.mcmc_covs, self.mcmc_weights, jitter=float(self.params.get("geom_jitter", 1e-6))
        )
        self.geom_nu = jnp.asarray(self.params.get("geom_nu", 5.0), dtype=self.geom_mu.dtype)

        
        self.target_fn = self.target_normal




    # ------------------------------------------------------------------------
    # Primitive dummy without a flow
    # -------------------------------------------------------------------------
    def attach_flow_and_scaler(self, *, flow, scaler_cfg, scaler_masks):
        """cannot provide these objects in __init__ so we are calling dummy"""
        self.flow = flow
        self.scaler_cfg = scaler_cfg
        self.scaler_masks = scaler_masks

    def run_experiment(self):
        sampler = self.params.get("sampler", "precond_pcn")

        if self.params["experiment_type"] == "gaussian" and sampler == "precond_pcn":
            self._run_preconditioned_pcn_gaussian()
            return

        raise ValueError(
            f"Unsupported combination experiment_type={self.params['experiment_type']} sampler={sampler}"
        )



    # -------------------------------------------------------------------------
    # Run pcn algorithm
    # -------------------------------------------------------------------------
    def _run_preconditioned_pcn_gaussian(self):
        # --- required objects ---
        if self.flow is None:
            # raise error
            print("Warning: self.flow is None; using IdentityFlow() for wiring test.")
            self.flow = IdentityFlow()

        if self.scaler_cfg is None or self.scaler_masks is None:
            raise ValueError(
                "scaler_cfg / scaler_masks are required for inverse_jax/forward_jax. "
                )

        D = int(self.params["n_dims"])
        N = int(self.params.get("n_walkers", 2048))

        # outer iterations: each call to preconditioned_pcn_jax adapts sigma/mu internally up to n_max
        n_outer = int(self.params.get("n_outer", 50))

        # kernel parameters
        beta = jnp.asarray(self.params.get("beta", 1.0), dtype=jnp.float32)
        n_max = int(self.params.get("n_max", 2000))
        n_steps = int(self.params.get("n_steps", 100))
        proposal_scale = jnp.asarray(self.params.get("proposal_scale", 0.2), dtype=jnp.float32)

        seed = int(self.params.get("seed", 0))
        key = jax.random.PRNGKey(seed)

        # prior / likelihood functions 
        logprior_fn = make_uniform_box_logprior(self.prior_low, self.prior_high)
        blob0 = jnp.zeros((0,), dtype=jnp.float32)

        def loglike_fn(xi):
            ll = self.likelihood.log_prob(xi)  # scalar
            return ll, blob0

        # -----------------------------
        # initialize ensemble state
        # -----------------------------
        key, k_init = jax.random.split(key, 2)
        u = jax.random.normal(k_init, shape=(N, D), dtype=jnp.float32)

        x, logdetj = inverse_jax(u, self.scaler_cfg, self.scaler_masks)

        # keep boundary-conditions consistent with kernel
        x_bc = apply_boundary_conditions_x_jax(x, dict(self.scaler_cfg))
        u_bc = forward_jax(x_bc, self.scaler_cfg, self.scaler_masks)
        x, logdetj = inverse_jax(u_bc, self.scaler_cfg, self.scaler_masks)
        u = u_bc

        finite0 = jnp.isfinite(logdetj) & jnp.all(jnp.isfinite(x), axis=1)

        def _prior_or_neginf(xi, ok):
            return jax.lax.cond(
                ok,
                lambda z: logprior_fn(z),
                lambda z: jnp.asarray(-jnp.inf, dtype=xi.dtype),
                xi,
            )

        logp = jax.vmap(_prior_or_neginf, in_axes=(0, 0), out_axes=0)(x, finite0)
        finite1 = finite0 & jnp.isfinite(logp)

        def _like_or_neginf(xi, ok):
            def _do(z):
                return loglike_fn(z)
            def _skip(z):
                return jnp.asarray(-jnp.inf, dtype=xi.dtype), blob0
            return jax.lax.cond(ok, _do, _skip, xi)

        logl, _ = jax.vmap(_like_or_neginf, in_axes=(0, 0), out_axes=(0, 0))(x, finite1)

        # recomputes logdetj_flow internally 
        logdetj_flow = jnp.zeros((N,), dtype=jnp.float32)
        blobs = jnp.zeros((N, 0), dtype=jnp.float32)

        # store data
        xs = []
        accept_hist = []
        sigma_hist = []
        calls_hist = []

        # -----------------------------
        # Outer loop: accumulate samples
        # -----------------------------
        for t in range(n_outer):
            out = preconditioned_pcn_jax(
                key,
                u=u,
                x=x,
                logdetj=logdetj,
                logl=logl,
                logp=logp,
                logdetj_flow=logdetj_flow,
                blobs=blobs,
                beta=beta,
                loglike_fn=loglike_fn,
                logprior_fn=logprior_fn,
                flow=self.flow,
                scaler_cfg=self.scaler_cfg,
                scaler_masks=self.scaler_masks,
                geom_mu=self.geom_mu,
                geom_cov=self.geom_cov,
                geom_nu=self.geom_nu,
                n_max=n_max,
                n_steps=n_steps,
                proposal_scale=proposal_scale,
                condition=None,
            )

            # update state
            key = out["key"]
            u = out["u"]
            x = out["x"]
            logdetj = out["logdetj"]
            logdetj_flow = out["logdetj_flow"]
            logl = out["logl"]
            logp = out["logp"]
            blobs = out["blobs"]

            xs.append(x)
            accept_hist.append(out["accept"])
            sigma_hist.append(out["proposal_scale"])
            calls_hist.append(out["calls"])

            if (t + 1) % int(self.params.get("print_every", 10)) == 0:
                acc = float(np.asarray(out["accept"]))
                sig = float(np.asarray(out["proposal_scale"]))
                calls = int(np.asarray(out["calls"]))
                steps = int(np.asarray(out["steps"]))
                print(f"[outer {t+1:>4d}/{n_outer}] accept={acc:.4f} sigma={sig:.4f} calls={calls} steps={steps}")

        # store results
        self.samples = np.asarray(jnp.stack(xs, axis=0))  # (n_outer, N, D)
        self.accept_history = np.asarray(jnp.stack(accept_hist))
        self.sigma_history = np.asarray(jnp.stack(sigma_hist))
        self.calls_history = np.asarray(jnp.stack(calls_hist))

        # print summary
        print(
            f"Done. samples shape={self.samples.shape} "
            f"mean_accept={self.accept_history.mean():.4f} "
            f"last_sigma={self.sigma_history[-1]:.4f}"
        )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def target_normal(self, x, data=None):
        return self.likelihood.log_prob(x)

    def get_next_available_outdir(self, base_dir: str, prefix: str = "results") -> str:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        matches = [re.match(rf"{prefix}_(\d+)", name) for name in existing]
        numbers = [int(m.group(1)) for m in matches if m]
        next_number = max(numbers, default=0) + 1
        return os.path.join(base_dir, f"{prefix}_{next_number}")

    @staticmethod
    def make_auto_bounds_inflated(means, covs, inflate=9.0, nsig=12.0, pad=1e-6,
                                 prior_low=None, prior_high=None):
        means = np.asarray(means, dtype=float)                 # (K, D)
        covs = np.asarray(covs, dtype=float) * float(inflate)  # 

        mu_min = means.min(axis=0)                             # (D,)
        mu_max = means.max(axis=0)                             # (D,)

        std_max = np.sqrt(np.stack([np.diag(C) for C in covs], axis=0)).max(axis=0)  # (D,)

        low = mu_min - nsig * std_max - pad
        high = mu_max + nsig * std_max + pad

        if prior_low is not None:
            low = np.minimum(low, float(prior_low))
        if prior_high is not None:
            high = np.maximum(high, float(prior_high))
        return low, high




    def get_true_and_mcmc_samples(self, discard=0, thin=1):
        dim = int(self.params["n_dims"])

        if not hasattr(self, "true_samples") or self.true_samples is None:
            raise ValueError("No true samples found. Ensure self.true_samples is set (gaussian experiment).")

        true_np = np.asarray(self.true_samples).reshape(-1, dim)

        if hasattr(self, "samples") and self.samples is not None:
            samp = np.asarray(self.samples).reshape(-1, dim)  # works for (n_outer, N, D) 
            samp = samp[int(discard)::int(thin), :]
            mcmc_np = samp
        else:
            raise ValueError("No sampler samples found. Run run_experiment() first.")

        return true_np, mcmc_np
    


    def plot_true_vs_mcmc_corner(self, seed=2046):
        """
        Overlay corner plot:
        - MCMC production samples (black)
        - true samples (red)
        Saves: true_vs_mcmc_corner_plot.pdf
        """
        # Get samples 
        true_np, mcmc_np = self.get_true_and_mcmc_samples()

        dim = int(self.params["n_dims"])
        labels = [f"x{i}" for i in range(dim)]

        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        # Plot MCMC first 
        fig = corner.corner(
            mcmc_np,
            color="black",
            hist_kwargs={"color": "black", "density": True},
            show_titles=True,
            labels=labels,
        )

        # Overlay true samples 
        corner.corner(
            true_np,
            fig=fig,
            color="red",
            hist_kwargs={"color": "red", "density": True},
            show_titles=True,
            labels=labels,
        )

        # Legend
        handles = [
            plt.Line2D([], [], color="black", label="pocomc"),
            plt.Line2D([], [], color="red", label="True Normal"),
        ]
        fig.legend(handles=handles, loc="upper right")

        save_name = os.path.join(outdir, "true_vs_mcmc_corner_plot.pdf")
        fig.savefig(save_name, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved overlay corner plot to {save_name}")



    def plot_acceptance_rate(self):
        print("Plotting acceptance-rate diagnostic curve...")

        if self.accept_history is None:
            raise ValueError("No accept_history found. Run run_experiment() first.")

        accept = np.asarray(self.accept_history).reshape(-1)

        plt.figure(figsize=(6, 4))
        plt.plot(accept)
        plt.xlabel("Outer iteration")
        plt.ylabel("Mean acceptance (alpha)")
        plt.title("Preconditioned pCN Acceptance")
        save_name = os.path.join(self.params["outdir"], "acceptance_rate_curve.pdf")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        print(f"Saved to {save_name}")



    def plot_sigma(self):
        if self.sigma_history is None:
            raise ValueError("No sigma_history found. Run run_experiment() first.")

        sig = np.asarray(self.sigma_history).reshape(-1)
        plt.figure(figsize=(6, 4))
        plt.plot(sig)
        plt.xlabel("Outer iteration")
        plt.ylabel("proposal_scale (sigma)")
        plt.title("Preconditioned pCN Sigma Adaptation")
        save_name = os.path.join(self.params["outdir"], "sigma_curve.pdf")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        print(f"Saved to {save_name}")





    #-----------------------------------------------------------------------------
    # 3.3. SAMPLE STATISTICS
    #-----------------------------------------------------------------------------
    def save_samples_json(self):
        # output directory 
        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        # get samples once
        true_np, mcmc_np = self.get_true_and_mcmc_samples()

        # save generated samples
        mcmc_path = os.path.join(outdir, "mcmc_samples.json")
        with open(mcmc_path, "w", encoding="utf-8") as f:
            json.dump(mcmc_np.tolist(), f)
        print(f"MCMC samples saved to {mcmc_path}")

        # save true samples
        true_path = os.path.join(outdir, "true_samples.json")
        with open(true_path, "w", encoding="utf-8") as f:
            json.dump(true_np.tolist(), f)
        print(f"True samples saved to {true_path}")




    def compute_and_save_sample_statistics(self):
        """
        Computes and saves per-dimension statistics for:
        - MCMC production samples
        - true samples
        Saves: sample_statistics.txt in self.params["outdir"]
        """

        # get samples 
        true_samples, mcmc_samples = self.get_true_and_mcmc_samples()

        # MCMC stats
        self.pm = mcmc_samples.mean(axis=0)
        self.pv = mcmc_samples.var(axis=0)
        self.ps = mcmc_samples.std(axis=0)

        # True stats
        self.qm = true_samples.mean(axis=0)
        self.qv = true_samples.var(axis=0)
        self.qs = true_samples.std(axis=0)

        # store arrays 
        self.mcmc_samples = mcmc_samples
        self.true_samples_np = true_samples

        np.set_printoptions(precision=4, suppress=True)

        stats_str = (
            "pm (mean of MCMC samples):\n" + str(self.pm) +
            "\n\npv (variance of MCMC samples):\n" + str(self.pv) +
            "\n\nps (std dev of MCMC samples):\n" + str(self.ps) +
            "\n\nqm (mean of true samples):\n" + str(self.qm) +
            "\n\nqv (variance of true samples):\n" + str(self.qv) +
            "\n\nqs (std dev of true samples):\n" + str(self.qs) + "\n"
        )

        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        stats_path = os.path.join(outdir, "sample_statistics.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(stats_str)

        print(f"Sample statistics saved to {stats_path}")



    #-----------------------------------------------------------------------------
    # 3.4. KL DIVERGENCE
    #-----------------------------------------------------------------------------
    import numpy as np, warnings, os
    from typing import Tuple


    @staticmethod
    def gau_kl(pm: np.ndarray, pv: np.ndarray,
               qm: np.ndarray, qv: np.ndarray) -> float:
        """
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Also computes KL divergence from a single Gaussian pm,pv to a set
         of Gaussians qm,qv.
        Diagonal covariances are assumed.  Divergence is expressed in nats.
        """
        if (len(qm.shape) == 2):
            axis = 1
        else:
            axis = 0
        # Determinants of diagonal covariances pv, qv
        dpv = pv.prod()
        dqv = qv.prod(axis)
        # Inverse of diagonal covariance qv
        iqv = 1. / qv
        # Difference between means pm, qm
        diff = qm - pm
        return (0.5 * (
            np.log(dqv / dpv)                 # log |\Sigma_q| / |\Sigma_p|
            + (iqv * pv).sum(axis)            # + tr(\Sigma_q^{-1} * \Sigma_p)
            + (diff * iqv * diff).sum(axis)   # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
            - len(pm)                         # - N
        ))
    

    def kl_metrics(
        self,
        outdir: str | None = None,
        filename: str = "kl_metrics.txt",
    ) -> None:
        import os
        import numpy as np

        # define outdir
        outdir = (
            outdir
            or (getattr(self, "params", {}) or {}).get("outdir", None)
            or getattr(self, "outdir", None)
        )
        if outdir is None:
            raise ValueError("No output directory specified (pass outdir=... or set params['outdir']).")
        os.makedirs(outdir, exist_ok=True)

        
        true_np, mcmc_np = self.get_true_and_mcmc_samples() 

        # Parametric Gaussian stats (diagonal covariance assumed)
        pm = mcmc_np.mean(axis=0)
        pv = mcmc_np.var(axis=0)
        qm = true_np.mean(axis=0)
        qv = true_np.var(axis=0)

        kl_val = self.gau_kl(pm, pv, qm, qv)  # scalar for 1D qm/qv

        out_path = os.path.join(outdir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            if np.isscalar(kl_val):
                f.write(f"Parametric KL (Gaussian): {float(kl_val):.8f}\n")
            else:
                kl_arr = np.asarray(kl_val).ravel()
                f.write("Parametric KL (Gaussian):\n")
                for i, v in enumerate(kl_arr):
                    f.write(f"  [{i}] {float(v):.8f}\n")

        print(f"KL metrics saved to {out_path}")








# Dummy mapping just to run experiment without a flow

import jax.numpy as jnp

# Identity "scaler": u <-> x
def inverse_jax(u, scaler_cfg=None, scaler_masks=None):
    u = jnp.asarray(u)
    logdet = jnp.zeros((u.shape[0],), dtype=u.dtype)  # (N,)
    return u, logdet

def forward_jax(x, scaler_cfg=None, scaler_masks=None):
    return jnp.asarray(x)

def apply_boundary_conditions_x_jax(x, cfg_dict=None):
    return jnp.asarray(x)

flow = IdentityFlow()
scaler_cfg = {}      # empty mapping 
scaler_masks = {}







##########################################################################################
# RUN EXPERIMENT
##########################################################################################


from types import SimpleNamespace

args = SimpleNamespace(
    outdir="./results",
    experiment_type="gaussian",

    n_dims=2,
    nr_of_components=2,
    nr_of_samples=10000,
    width_mean=10.0,
    width_cov=1.0,
    weights_of_components=None,

    sampler="precond_pcn",
    n_walkers=400,        # start smaller for a quick test
    n_outer=100,
    n_max=1000,
    n_steps=300,
    proposal_scale=0.2,
    beta=1.0,
    seed=55,
    print_every=10,

    geom_nu=1,
    prior_inflate=16.0,
    prior_nsig=18.0,
    prior_pad=1e-6,
    geom_jitter=1e-6,
)



def main():
    # Get the arguments passed over from the command line, and create the experiment runner
    
    runner = pcn_ExperimentRunner(args, flow=flow, scaler_cfg=scaler_cfg, scaler_masks=scaler_masks)
    runner.run_experiment()
    runner.plot_true_vs_mcmc_corner()
    runner.plot_acceptance_rate()
    runner.plot_sigma()
    runner.save_samples_json()
    runner.compute_and_save_sample_statistics()
    runner.kl_metrics()

if __name__ == "__main__":
    main()