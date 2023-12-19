import shap
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesRegressor,
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesClassifier,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
    OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
    SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, PassiveAggressiveClassifier,
    TheilSenRegressor, LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron,
    LassoLarsIC, PoissonRegressor, GammaRegressor, TweedieRegressor)
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor, ExtraTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier 


class ArtifactBase():
    """
    ArtifactBase is a base class for logging various machine learning artifacts to MLflow.

    It provides utility methods for logging different types of artifacts, such as plots, reports, and data samples,
    to the MLflow tracking server. Subclasses can inherit from this base class and extend its functionality
    to log specific types of artifacts related to their machine learning workflows.

    Attributes:
        None

    Methods:
        - Various methods for logging different types of machine learning artifacts.

    Example:
    ```python
    # Import the base class
    from artifact_base import ArtifactBase

    # Create a subclass and extend its functionality
    class MyArtifactLogger(ArtifactBase):
        def __init__(self):
            super().__init__()

        def log_custom_artifact(self, custom_data):
            # Implement custom artifact logging logic here
            pass

    # Create an instance of the subclass
    artifact_logger = MyArtifactLogger()

    # Use the instance to log various machine learning artifacts
    artifact_logger.log_custom_artifact(custom_data)
    ```
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def get_shap_explainer(self, pipeline, model_step, X):
        """
        Determines and returns the appropriate SHAP (SHapley Additive exPlanations) explainer based on the model type.

        SHAP is a unified measure of feature importance that can be used to explain the output of any machine learning model.
        This method selects the appropriate SHAP explainer based on the type of model provided in the pipeline.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): An object implementing the "fit" and "predict" methods.

        model_step (str): The name of the step in the pipeline that represents the model.

        X (pd.DataFrame): The input features used for SHAP value calculation.

        Returns:
        tuple: A tuple containing a SHAP explainer object and the input feature DataFrame (X).

        Example:
        ```python
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        import shap

        # Create a pipeline with a RandomForestClassifier model
        model = Pipeline([
            ('classifier', RandomForestClassifier())
        ])

        # Generate example data
        X = pd.DataFrame({'Feature1': [1, 2, 3, 4, 5], 'Feature2': [2, 4, 5, 4, 5]})

        # Get the SHAP explainer
        explainer, X_transformed = ArtifactBase.get_shap_explainer(model, 'classifier', X)
        ```
        
        Notes:
        - This method checks the type of the model in the pipeline and selects the appropriate SHAP explainer:
          - For tree-based models (e.g., RandomForest, DecisionTree), it uses `shap.TreeExplainer`.
          - For linear models (e.g., LinearRegression, LogisticRegression), it uses `shap.LinearExplainer`.
          - For other models, it defaults to a `shap.KernelExplainer`.

        - If a transformation step is present before the model, it applies the transformation to the input features (X) before calculating SHAP values.

        - The returned tuple contains the SHAP explainer object and the transformed input features (X), if applicable.

        Raises:
        - None

        Returns:
        - A tuple containing a SHAP explainer object and the input feature DataFrame (X).
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
                       ExtraTreeRegressor, ExtraTreeClassifier,
                       XGBClassifier, XGBRegressor, LGBMRegressor,
                       LGBMClassifier, HistGradientBoostingClassifier,
                       HistGradientBoostingRegressor)):
            return shap.TreeExplainer(model), X

        # Linear models
        elif isinstance(model,
                        (LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
                        OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
                        SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, PassiveAggressiveClassifier,
                        TheilSenRegressor, LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron,
                        LassoLarsIC, PoissonRegressor, GammaRegressor, TweedieRegressor)):
            return shap.LinearExplainer(model, X), X

        else:
            # Default to Explainer for models not explicitly handled above
            if hasattr(model, 'predict_proba'):
                return shap.KernelExplainer(model.predict_proba,
                                            X), X
            else:
                return shap.KernelExplainer(model.predict,
                                            X), X
