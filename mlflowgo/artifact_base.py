import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesRegressor
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, Lars,
    LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
    SGDRegressor, PassiveAggressiveRegressor)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class ArtifactBase():

    def __init__(self, **kwargs) -> None:
        self.pipeline = kwargs.get('pipeline', None)
        self.X_train = kwargs.get('X_train', None)
        self.y_train = kwargs.get('y_train', None)
        self.X_test = kwargs.get('X_test', None)
        self.y_test = kwargs.get('y_test', None)
        self.model_step = kwargs.get('model_step', None)
        self.feature_names = kwargs.get('feature_names', None)
        self.metric = kwargs.get('metric', None)
        self.param_name = kwargs.get('param_name', None)
        self.param_range = kwargs.get('param_range', None)
        self.objective = kwargs.get('objective', None)
        self.dataset_desc = kwargs.get('dataset_desc', None)

    @classmethod
    def get_shap_explainer(self, model, X):
        """
        Determines and returns the appropriate SHAP explainer based on the model type.

        Parameters:
        model: The trained machine learning model.
        X (pd.DataFrame): The input features used for SHAP value calculation.

        Returns:
        A SHAP explainer object.
        """

        # Tree-based models
        if isinstance(model,
                      (RandomForestClassifier, GradientBoostingClassifier,
                       DecisionTreeClassifier, DecisionTreeRegressor,
                       ExtraTreesRegressor)):
            return shap.TreeExplainer(model)

        # Linear models
        elif isinstance(model,
                        (LogisticRegression, LinearRegression, Ridge,
                         Lasso, ElasticNet, Lars, LassoLars,
                         OrthogonalMatchingPursuit, BayesianRidge,
                         ARDRegression, SGDRegressor, PassiveAggressiveRegressor)):
            return shap.LinearExplainer(model, X)

        elif isinstance(model, KNeighborsRegressor):
            return shap.Explainer(model.predict, X)

        else:
            # Default to Explainer for models not explicitly handled above
            if hasattr(model, 'predict_proba'):
                return shap.KernelExplainer(model.predict_proba, X)
            else:
                return shap.KernelExplainer(model, X)

            #return shap.Explainer(model, X)
