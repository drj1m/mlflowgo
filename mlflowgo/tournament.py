from .base import Base
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import numpy as np
from sklearn.base import is_classifier
from . import CLASSIFIER_KEY, REGRESSOR_KEY
import sklearn.metrics as sklm


class Tournament(Base):
    """
    Tournament class for machine learning model evaluation and selection.

    This class extends the `Base` class and provides functionality for running machine learning pipelines,
    selecting top-performing models, and managing model information and parameters.

    Parameters:
        X_train (pd.DataFrame, optional): The training feature data (default is None).
        y_train (pd.Series, optional): The training target data (default is None).
        X_test (pd.DataFrame, optional): The test feature data (default is None).
        y_test (pd.Series, optional): The test target data (default is None).
        pipelines (list or None, optional): A list of machine learning pipelines or None to auto-generate (default is None).
        feature_names (list, optional): List of feature names (default is derived from X_train columns).
        metrics (dict or None, optional): Dictionary of metrics for model evaluation (default is None).
        param_name (str, optional): The name of the hyperparameter to tune (default is None).
        param_range (list, optional): The range of hyperparameter values for tuning (default is None).
        objective (str, optional): The objective of the experiment (default is None).
        dataset_desc (str, optional): Description of the dataset (default is None).

    Attributes:
        models (dict): Dictionary to store machine learning models with model names as keys.
        models_params (dict): Dictionary to store model parameters with model names as keys.
        final_scores (dict): Dictionary to store final model evaluation scores.
        model_info (dict): Dictionary to store model information.
        pipelines (list): List of machine learning pipelines.
        feature_names (list): List of feature names.
        metrics (dict): Dictionary of metrics for model evaluation.
        param_name (str): The name of the hyperparameter to tune.
        param_range (list): The range of hyperparameter values for tuning.
        objective (str): The objective of the experiment.
        dataset_desc (str): Description of the dataset.
    """
    def __init__(self, **kwargs):
        """
        Initialize a Tournament instance.

        Initializes the Tournament instance with optional parameters for training and testing data,
        pipelines, and other experiment-specific information.

        Args:
            **kwargs: Keyword arguments for specifying instance parameters.

        Returns:
            None
        """
        self.models = {}
        self.models_params = {}
        self.final_scores = {}
        self.model_info = {}
        self.X_train = kwargs.get('X_train', None)
        self.y_train = kwargs.get('y_train', None)
        self.X_test = kwargs.get('X_test', None)
        self.y_test = kwargs.get('y_test', None)
        self.pipelines = kwargs.get('pipelines', None)
        self.feature_names = kwargs.get('feature_names', self.X_train.columns)
        self.metrics = kwargs.get('metrics', None)
        self.param_name = kwargs.get('param_name', None)
        self.param_range = kwargs.get('param_range', None)
        self.objective = kwargs.get('objective', None)
        self.dataset_desc = kwargs.get('dataset_desc', None)    

    @property
    def pipelines(self):
        return self._pipelines

    @pipelines.setter
    def pipelines(self, value):
        pipelines = value if value is not None else self._find_best_models()
        if not isinstance(pipelines, (list, tuple, np.ndarray)):
            self._pipelines = [pipelines]
        else:
            self._pipelines = pipelines
        self._pipeline_array_to_dict()

    @property
    def task_type(self):
        return CLASSIFIER_KEY if is_classifier(self.pipeline) else REGRESSOR_KEY

    def run_name(self, pipeline):
        """
        Return the run name based on the given pipeline.

        Parameters:
            pipeline (Pipeline): The pipeline for which the run name is generated.

        Returns:
            str: A string representing the generated run name.
        """
        return self.get_run_name(pipeline)

    def run(self, run_id, pipeline, cv, grid_search=False):
        """
        Run a machine learning pipeline with optional grid search and cross-validation.

        Parameters:
            run_id (str): An identifier for the current run.
            pipeline (Pipeline): The machine learning pipeline to be executed.
            cv (int): The number of cross-validation folds. Use -1 for no cross-validation.
            grid_search (bool, optional): Whether to perform grid search for hyperparameter tuning (default is False).
        """
        self.pipeline = pipeline
        self.metrics = self.get_model_metrics(
            self.metrics,
            self.task_type)
        self.model_step = self.get_model_step_from_pipeline(pipeline)
        self.model_name = pipeline.named_steps[self.model_step].__class__.__name__
        # Perform randomised grid search of params
        if grid_search:
            search = RandomizedSearchCV(
                self.pipeline,
                param_distributions=self.models_params[self.model_name],
                n_iter=20,
                cv=cv,
                scoring='neg_mean_squared_error',
                verbose=3
            )
            search.fit(self.X_train, self.y_train)
            self.pipeline = search.best_estimator_
        else:
            self.pipeline.fit(self.X_train, self.y_train)

        # Perform cross-validation
        if cv != -1:
            cv_results = cross_val_score(
                self.pipeline,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring='neg_mean_squared_error',
                verbose=3)
            self.final_scores[self.model_name] = np.mean(cv_results)
        else:
            cv_results = None
            self.final_scores[self.model_name] = sklm.mean_squared_error(
                self.y_test,
                np.nan_to_num(self.pipeline.predict(self.X_test))
            )

        self.model_info[self.model_name] = (run_id, self.model_name)

    def _find_best_models(self):
        """
        Find and return the top-performing machine learning models for the dataset.

        Returns:
            List[Pipeline]: A list of the top-performing machine learning pipelines.
        """
        final_models = []
        top_n = 5
        if len(np.unique(self.y_train)) / len(self.y_train) < 0.2:
            lp = LazyClassifier(predictions=True)
            _metric = 'F1 Score'
        else:
            lp = LazyRegressor(predictions=True)
            _metric = 'RMSE'
        models, predictions = lp.fit(
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        )
        top_models = models.sort_values(by=_metric).index.tolist()

        for _model in top_models:
            if len(final_models) > top_n:
                break
            _pipeline = self.get_basic_pipeline(_model)
            if _pipeline is not None:
                final_models.append(_pipeline)

        return final_models

    def _pipeline_array_to_dict(self):
        """
        Convert an array of machine learning pipelines into dictionaries for models and parameters.

        This method iterates over an array of machine learning pipelines and converts them into
        dictionaries where keys represent model names and values are the corresponding pipelines.
        Additionally, parameter dictionaries for each model are created and stored.

        Returns:
            None

        For each pipeline in the `pipelines` array, the method performs the following steps:
        1. Retrieves the model step from the pipeline.
        2. Obtains the model's class name.
        3. Fetches the parameter distribution for the model using the `get_param_dist` method.
        4. Stores the model-pipeline mapping in the `models` dictionary.
        5. If model parameters are available, they are converted into a dictionary with modified keys
        to include the model step and stored in the `models_params` dictionary.
        """
        for pipeline in self.pipelines:
            model_step = self.get_model_step_from_pipeline(pipeline)
            model_name = pipeline.named_steps[model_step].__class__.__name__
            model_param = self.get_param_dist(model_name)

            self.models[model_name] = pipeline
            if model_param is not None:
                model_param = {f'{model_step}__{i}': j for i, j in model_param.items()}
            self.models_params[model_name] = model_param
