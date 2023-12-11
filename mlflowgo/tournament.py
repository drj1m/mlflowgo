from .base import Base
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np


class Tournament(Base):
    """
    A class to run a tournament style selection to finding the best model, logging to MLflow
    """

    def __init__(self, X_train, y_train, X_test, y_test, pipelines=None):

        self.X_train = X_train
        self.y_train = y_train
        self.pipelines = pipelines
        self.models = {}
        self.models_params = {}

    def run(self):
        self._pipeline_array_to_dict()
        self._run_cv_param_search()

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
