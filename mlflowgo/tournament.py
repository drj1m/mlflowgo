from .base import Base
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import numpy as np


class Tournament(Base):
    """
    A class to run a tournament style selection to finding the best model, logging to MLflow
    """

    def __init__(self, X, y, pipelines=None):

        self.X = X
        self.y = y
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
            if model_param is not None:
                self.models[model_name] = pipeline
                model_param = {f'{model_step}__{i}':j for i, j in model_param.items()}
                self.models_params[model_name] = model_param

    def _run_cv_param_search(self):
        """
        Run cv on models with hyper-parameter tuning
        """
        n_iter_search = 20
        final_scores = {}

        for model_name in self.models:
            search = RandomizedSearchCV(
                self.models[model_name],
                param_distributions=self.models_params[model_name],
                n_iter=n_iter_search,
                cv=5,
                scoring='neg_mean_squared_error',
                verbose=3
            )
            search.fit(self.X, self.y)
            best_model = search.best_estimator_
            cv_scores = cross_val_score(
                best_model,
                self.X,
                self.y,
                cv=5,
                scoring='neg_mean_squared_error',
                verbose=3
            )
            final_scores[model_name] = np.mean(cv_scores)
            best_model_name = min(final_scores, key=final_scores.get)
        print(f'best model is {best_model_name}')
