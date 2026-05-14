# Model families

All models share the same data path (NWB → tensors). Most families use the shared `output_head` readout (default `log_link`); `ndt_lite` emits rates directly. They differ in how they map held-in spike counts to held-in and held-out neuron rates.

| Family | File | Key idea |
|---|---|---|
| `smoothing` | `models/smoothing.py` | Causal Gaussian smoothing of held-in counts, then a Poisson GLM onto held-out neurons. Oldest baseline; no latent structure. |
| `pca_latent_regression` | `models/pca_latent_regression.py` | Reduce held-in counts to a low-rank PCA latent (one time bin), then ridge-regress to held-out neurons. Tests whether a static latent is enough. |
| `ridge_direct` | `models/ridge_direct.py` | Direct ridge regression from current-bin held-in counts to held-out counts. Simplest possible baseline; no latent, no history. |
| `lagged_ridge_direct` | `models/lagged_ridge_direct.py` | Ridge regression from a short history window of held-in counts to held-out counts. Same as `ridge_direct` but with temporal context. |
| `lagged_pca_latent_regression` | `models/lagged_pca_latent_regression.py` | Lag the held-in counts, compress with PCA on the lagged feature matrix, then ridge-regress. Strongest validated model in this repo. |
| `lagged_reduced_rank_regression` | `models/lagged_reduced_rank_regression.py` | Supervised low-rank analogue of lagged PCA — ridge regression with a rank constraint on the coefficient matrix. Control for "is the PCA bottleneck doing real work?". |
| `lds_pca_latent_regression` | `models/lds_pca_latent_regression.py` | PCA latent fit jointly with a linear dynamical system (Kalman + RTS smoother) over time, then ridge to held-out neurons. Exploratory; compares against the non-dynamical lagged baseline. |
| `ndt_lite` | `models/ndt_lite.py` | Small PyTorch temporal transformer over held-in spike counts with random held-in masking, Poisson rate losses, optional neuron event embeddings, optional cosine learning-rate scheduling, and optional seed ensembling (members use seeds `seed`, `seed+1`, …; rates are averaged). First neural-sequence baseline toward NDT/STNDT-class methods. |
| `ndt_factorized` | `models/ndt_factorized.py` | Experimental neuron-aware transformer: held-in neurons are embedded as tokens, compressed into learned per-time latent tokens, modeled over time, and decoded with a global temporal head plus neuron-aware residual readout. The first bounded mc_maze run underperformed NDT-lite, so this is a foundation/guardrail rather than a headline model. |

## Design principles

- **One `fit_predict_*` per family.** Every model exposes a single callable that takes the built tensors and returns held-in + held-out rate predictions. No hidden state across calls.
- **Train-only preprocessing.** Any scaling / PCA basis / lag-buffer initial values are fit on train data only, then applied frozen to eval tensors. This is enforced in `models/temporal_features.py`.
- **Shared rate readout for linear baselines.** Families registered with `uses_rate_head=True` route predictions through `models/output_head.py` so cross-model comparisons stay apples-to-apples (see `docs/output_head_postmortem.md`). The `ndt_lite` family sets `uses_rate_head=False` and emits positive Poisson rates directly from the network.
- **Neural baselines stay optional.** `ndt_lite` and `ndt_factorized` require PyTorch and are installed with `pip install -e .[neural]`; the default linear/reproducibility path remains lightweight.
- **Declarative registration.** Each family is registered in `src/nlb_project/model_registry.py` with its parameter names, sweep axes, and CV defaults. Adding a new family means writing the model file and appending one `ModelSpec` entry.
