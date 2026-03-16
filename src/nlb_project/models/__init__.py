"""Model implementations for NLB project."""

from .pca_latent_regression import predict_pca_latent_regression
from .ridge_direct import predict_ridge_direct

__all__ = ["predict_pca_latent_regression", "predict_ridge_direct"]
