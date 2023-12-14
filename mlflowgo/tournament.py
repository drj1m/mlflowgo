from .base import Base
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.base import is_classifier
from . import CLASSIFIER_KEY, REGRESSOR_KEY
import sklearn.metrics as sklm
import mlflow


class Tournament(Base):
    """
    A class to run a tournament style selection to finding the best model, logging to MLflow
    """

    def __init__(self, **kwargs):

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
                self.pipeline.predict(self.X_test)
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
