import numpy as np
from typing import Union

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
            raise ValueError("model is constant, se X=None.")
        elif self.coef_ is None and X is None:
            return self.intercept_
        else:
            return super().predict(X)


def fit_controlled_linear_regression(X: Union[np.ndarray, None], Y: np.ndarray, Z: Union[np.ndarray, None], fit_intercept: bool = True) -> LinearPredictionModel:    
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