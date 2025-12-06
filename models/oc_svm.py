from    typing                  import  Self

import  numpy                   as      np
from    sklearn.pipeline        import  Pipeline
from    sklearn.preprocessing   import  StandardScaler
from    sklearn.decomposition   import  PCA
from    sklearn.svm             import  OneClassSVM


class OC_SVM():
    def __init__(
            self,
            nu:             float   = 0.05,
            kernel:         str     = 'rbf',
            gamma:          str     = 'scale',
            n_components:   int     = 128,
        ) -> Self:
        """The initializer for `OC_SVM`.

        Arguments:
            `nu` (`float`): An upper bound of the fraction on the outliers. Defaults to 0.05.
            `kernel` (`str`): Specifies the kernel type to be used in the algorithm. Defaults to 'rbf'.
            `gamma` (`str`): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. Defaults to 'scale'.
            `n_components` (`int`): Number of components for PCA. Defaults to 128.
        """
        self.model = Pipeline(
            [
                ('scaler',  StandardScaler()),
                ('pca',     PCA(n_components=n_components)),
                ('oc_svm',  OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)),
            ]
        )
        return
    

    def fit(self, X: np.ndarray) -> None:
        """Fit the one-class SVM model according to the given training data."""
        self.model.fit(X)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    def decision_function(self, X: np.ndarray, inlier_as_zero: bool=True) -> np.ndarray:
        """The decision function for the One-Class SVM model. If `inlier_as_zero` is `True`, the decision values are negated so that inliers have negative values and outliers have positive values. Otherwise, the original decision values are returned."""
        base_decision_values = self.model.decision_function(X)
        if inlier_as_zero:  return -base_decision_values
        else:               return  base_decision_values