import shap
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesRegressor,
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesClassifier)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, Lars,
    LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
    SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, TheilSenRegressor,
    RidgeClassifier, SGDClassifier, Perceptron)
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor, ExtraTreeClassifier


class ArtifactBase():

    def __init__(self) -> None:
        pass

    @classmethod
    def get_shap_explainer(self, pipeline, model_step, X):
        """
        Determines and returns the appropriate SHAP explainer based on the model type.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): Object type that implements the "fit" and "predict" methods
        model_step (str): Step name for the model
        X (pd.DataFrame): The input features used for SHAP value calculation.

        Returns:
        A SHAP explainer object.
        """

        model = pipeline.named_steps[model_step]
        transform_pipeline = Pipeline(
            [step for step in pipeline.steps if step[0] != model_step]
        )

        if len(transform_pipeline) > 0:
            X = pd.DataFrame(data=transform_pipeline.transform(X),
                             columns=transform_pipeline.get_feature_names_out())

        # Tree-based models
        if isinstance(model,
                      (RandomForestClassifier, GradientBoostingClassifier,
                       DecisionTreeClassifier, DecisionTreeRegressor,
                       ExtraTreesRegressor, RandomForestRegressor,
                       GradientBoostingRegressor, ExtraTreesClassifier,
                       ExtraTreeRegressor, ExtraTreeClassifier)):
            return shap.TreeExplainer(model), X

        # Linear models
        elif isinstance(model,
                        (LogisticRegression, LinearRegression, Ridge,
                         Lasso, ElasticNet, Lars, LassoLars,
                         OrthogonalMatchingPursuit, BayesianRidge,
                         ARDRegression, SGDRegressor, PassiveAggressiveRegressor,
                         HuberRegressor, TheilSenRegressor, RidgeClassifier,
                         SGDClassifier, Perceptron)):
            return shap.LinearExplainer(model, X), X

        else:
            # Default to Explainer for models not explicitly handled above
            if hasattr(model, 'predict_proba'):
                return shap.KernelExplainer(model.predict_proba,
                                            X), X
            else:
                return shap.KernelExplainer(model.predict,
                                            X), X
