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
        return self.get_run_name(pipeline)

    def run(self, run_id, pipeline, cv, grid_search=False):
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
        """ Use lazypredict to find the best models to evaluate
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
        """ Converts a pipeline of models to dict
        """
        for pipeline in self.pipelines:
            model_step = self.get_model_step_from_pipeline(pipeline)
            model_name = pipeline.named_steps[model_step].__class__.__name__
            model_param = self.get_param_dist(model_name)

            self.models[model_name] = pipeline
            if model_param is not None:
                model_param = {f'{model_step}__{i}': j for i, j in model_param.items()}
            self.models_params[model_name] = model_param

    def _run_cv_param_search(self):
        """
        Run cv on models with hyper-parameter tuning
        """
        n_iter_search = 20
        final_scores = {}

        for model_name in self.models:
            if self.models_params[model_name] is not None:
                search = RandomizedSearchCV(
                    self.models[model_name],
                    param_distributions=self.models_params[model_name],
                    n_iter=n_iter_search,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    verbose=3
                )
                search.fit(self.X_train, self.y_train)
                best_model = search.best_estimator_
                cv_scores = cross_val_score(
                    best_model,
                    self.X_train,
                    self.y_train,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    verbose=3
                )
                final_scores[model_name] = np.mean(cv_scores)
            else:
                model = self.models[model_name]
                model.fit(self.X_train, self.y_train)
                cv_scores = cross_val_score(
                    model,
                    self.X_train,
                    self.y_train,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    verbose=3
                )
                final_scores[model_name] = np.mean(cv_scores)

        best_model_name = min(final_scores, key=final_scores.get)
        print(f'best model is {best_model_name}')
