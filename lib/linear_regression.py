import numpy as np
from typing import Union
import torch

from sklearn.linear_model._base import LinearModel, LinearRegression

class LinearPredictionModel(LinearModel):
    def __init__(self, coef=None, intercept=None):
        super().__init__()
        if coef is not None:
            coef = np.array(coef)
            if intercept is None:
                intercept = np.zeros(coef.shape[0])
            else:
                pass
        else:
            if intercept is None:
                raise ValueError("Provide at least one of coef and intercept")
            else:
                pass
        self.intercept_ = intercept
        self.coef_ = coef
        
        self.intercept = intercept
        self.coef = coef

    def fit(self, X, y):
        raise NotImplementedError("model is only for prediction")
    
    def predict(self, X):
        if self.coef_ is None and X is not None:
            raise ValueError("model is constant, set X = None.")
        elif self.coef_ is None and X is None:
            return self.intercept_
        else:
            return super().predict(X)


def _fit_controlled_linear_regression_numpy(X: Union[np.ndarray, None], Y: np.ndarray, Z: Union[np.ndarray, None], fit_intercept: bool = True) -> LinearPredictionModel:    
    # Note this has same logic as sklearn.linear_regression, i.e. fits multi-target to same features
    if Z is not None and Z.ndim == 1:
        Z = Z.reshape((-1, 1))
    if X is not None and X.ndim == 1:
        X = X.reshape((-1, 1))

    lmXZ = LinearRegression(fit_intercept=fit_intercept)
    if X is not None and Z is not None:
        XZ = np.concatenate([X, Z], axis=-1)
        lmXZ.fit(XZ, Y)
        coef_X = lmXZ.coef_[:, :X.shape[1]] if Y.ndim == 2 else lmXZ.coef_[:X.shape[1]]
        intercept = lmXZ.intercept_
    elif X is None and Z is not None:
        lmXZ.fit(Z, Y)
        coef_X = None
        intercept = lmXZ.intercept_
    elif X is not None and Z is None:
        lmXZ.fit(X, Y)
        coef_X = lmXZ.coef_
        intercept = lmXZ.intercept_
    else:
        coef_X = None
        if fit_intercept:
            intercept = Y.mean(axis=0) if Y.ndim == 2 else Y.mean()
        else:
            intercept = np.zeros(Y.shape[1]) if Y.ndim == 2 else 0.0

    lmX = LinearPredictionModel(coef=coef_X, intercept=intercept)
    return lmX


def fit_controlled_linear_regression(X: Union[np.ndarray, torch.Tensor, None], Y: Union[np.ndarray, torch.Tensor], Z: Union[np.ndarray, torch.Tensor, None], fit_intercept: bool = True, parallel: bool = False) -> LinearPredictionModel:    
    if all([isinstance(M, np.ndarray) or M is None for M in [X, Y, Z]]):
        if parallel:
            raise ValueError('parallel not supported for numpy arrays, current implementation uses scikit-learn which uses all rgressor across all targets.')
        return _fit_controlled_linear_regression_numpy(X, Y, Z, fit_intercept=fit_intercept)
    elif all([isinstance(M, torch.Tensor) or M is None for M in [X, Y, Z]]):
        return _fit_controlled_linear_regression_torch(X, Y, Z, fit_intercept=fit_intercept, parallel=parallel)
    else:
        raise ValueError('X, Y and Z must all be of the same type (or None): np.ndarray or torch.Tensor.')


class TorchLinearRegression(LinearRegression):
    def __init__(self, parallel: bool = False, fit_intercept: bool = True):
        self.parallel = parallel
        self.fit_intercept = fit_intercept

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        assert X.shape[0] == y.shape[0]
        if self.parallel:
            # X has shape (n_samples, n_features, n_targets) and y has shape (n_samples, n_targets)
            assert X.ndim == 3 and y.ndim == 2
            assert X.shape[-1] == y.shape[-1]
        else:
            # X has shape (n_samples, n_features) and y has shape (n_samples,) or (n_samples, n_targets) 
            assert X.ndim == 2

        if self.fit_intercept:
            if self.parallel:
                X = torch.cat([torch.ones(X.shape[0], 1, X.shape[-1]), X], dim=1)
            else:
                X = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)

        if self.parallel:
            # set n_targets as first dim
            X = X.permute(2, 0, 1)
            y = y.transpose(0, 1)

            # set non-invertible entries so that betas are set to zero (and intercept to mean)
            XtX = X.transpose(-1, -2) @ X
            mask = XtX.det() == 0
            XtX[mask] = torch.eye(XtX.shape[-1])
            X[mask] = torch.zeros_like(X[0])

            # beta has shape [n_targets, n_features (+ 1)]
            if self.fit_intercept:
                X[mask, 0] = 1.
            beta = torch.bmm(
                torch.inverse(XtX), 
                (X.transpose(-1, -2) @ y.unsqueeze(-1))
            ).squeeze(-1)
        else:
            beta = torch.inverse(X.T @ X) @ X.T @ y
            
            # beta has shape [n_features (+ 1)] or [n_targets, n_features (+ 1)]
            if y.ndim == 2:
                beta = beta.T

        if self.fit_intercept:
            if y.ndim == 2:
                self.intercept_ = beta[:, 0]
                self.coef_ = beta[:, 1:]
            else:
                self.intercept_ = beta[0]
                self.coef_ = beta[1:]
        else:
            self.coef_ = beta

        return self
    
    def predict(self, X: torch.Tensor):
        if self.parallel:
            assert X.ndim == 3 and self.coef_.ndim == 2
            # X has shape (n_samples, n_features, n_targets) and self.coef_ has shape (n_targets, n_features)
            X = X.permute(2, 0, 1)
            pred = (X @ self.coef_.unsqueeze(-1)).squeeze(-1)
            # pred has shape (n_samples, n_targets)
            pred = pred.transpose(0, 1)
        else:
            # X has shape (n_samples, n_features) and self.coef_ has shape (n_features,) or (n_targets, n_features)
            assert X.ndim == 2
            # pred has shape (n_samples, n_targets) or (n_samples,)
            pred = X @ self.coef_.T if self.coef_.ndim == 2 else X @ self.coef_
        if self.fit_intercept:
            # pred has shape (n_samples, n_targets) and self.intercept_ has shape (n_targets,) or pred has shape (n_samples,) and self.intercept_ is scalar
            return self.intercept_.unsqueeze(0) + pred if pred.ndim == 2 else self.intercept_ + pred
        else:
            return pred
    

class TorchLinearPredictionModel(LinearModel):
    def __init__(self, coef=None, intercept=None):
        super().__init__()
        if coef is not None:
            if not isinstance(coef, torch.Tensor):
                coef = torch.tensor(coef)
            if intercept is None:
                intercept = torch.zeros(coef.shape[0])
            else:
                pass
        else:
            if intercept is None:
                raise ValueError("Provide at least one of coef and intercept")
            else:
                pass
        self.intercept_ = intercept
        self.coef_ = coef
        
        self.intercept = intercept
        self.coef = coef

    def fit(self, X, y):
        raise NotImplementedError("model is only for prediction")
    
    def predict(self, X):
        if self.coef_ is None and X is not None:
            raise ValueError("model is constant, set X = None.")
        elif self.coef_ is None and X is None:
            return self.intercept_
        else:
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
            return self.intercept_ + X @ self.coef_.T if self.coef_.ndim == 2 else self.intercept_ + X @ self.coef_
        

def _fit_controlled_linear_regression_torch(X: Union[torch.Tensor, None], Y: torch.Tensor, Z: Union[torch.Tensor, None], fit_intercept: bool = True, parallel: bool = False) -> TorchLinearPredictionModel:    
    if Z is not None and Z.ndim == 1:
        Z = Z.reshape((-1, 1))
    if X is not None and X.ndim == 1:
        X = X.reshape((-1, 1))

    if parallel:
        assert Y.ndim == 2
        if X is not None:
            assert X.shape[0] == Y.shape[0] and X.shape[-1] == Y.shape[-1]
            if X.ndim == 2:
                X = X.unsqueeze(1)
        if Z is not None:
            assert Z.shape[0] == Y.shape[0] and Z.shape[-1] == Y.shape[-1]
            if Z.ndim == 2:
                Z = Z.unsqueeze(1)

    lmXZ = TorchLinearRegression(fit_intercept=fit_intercept, parallel=parallel)
    if X is not None and Z is not None:
        XZ = torch.cat([X, Z], dim=1)
        lmXZ.fit(XZ, Y)
        coef_X = lmXZ.coef_[:, :X.shape[1]] if Y.ndim == 2 else lmXZ.coef_[:X.shape[1]]
        intercept = lmXZ.intercept_
    elif X is None and Z is not None:
        lmXZ.fit(Z, Y)
        coef_X = None
        intercept = lmXZ.intercept_
    elif X is not None and Z is None:
        lmXZ.fit(X, Y)
        coef_X = lmXZ.coef_
        intercept = lmXZ.intercept_
    else:
        coef_X = None
        if fit_intercept:
            intercept = torch.mean(Y, dim=0) if Y.ndim == 2 else torch.mean(Y)
        else:
            intercept = torch.zeros(Y.shape[1]) if Y.ndim == 2 else 0.0

    lmX = TorchLinearPredictionModel(coef=coef_X, intercept=intercept)
    return lmX