import pytest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, make_friedman1
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
from mlflowgo.base import Base
from mlflowgo import REGRESSOR_KEY, CLASSIFIER_KEY


@pytest.fixture
def valid_pipeline():
    return Pipeline([
        ('pca', PCA(n_components=2)),
        ('rf', RandomForestClassifier())
    ])


@pytest.fixture
def invalid_pipeline():
    return Pipeline([
        ('pca', PCA(n_components=2))
    ])


@pytest.fixture
def no_model_pipeline():
    return Pipeline([
        ('pca', PCA(n_components=2))
    ])


@pytest.fixture
def iris():
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )
    return data


@pytest.fixture
def df():
    X, y = make_friedman1(n_samples=100, n_features=5, noise=0.1)
    feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df


def test_with_valid_pipeline(valid_pipeline):
    model_step = Base().get_model_step_from_pipeline(valid_pipeline)
    assert model_step == 'rf', "Failed to identify the model step in a valid pipeline."


def test_with_invalid_input(invalid_pipeline):
    with pytest.raises(ValueError):
        Base().get_model_step_from_pipeline("not a pipeline")


def test_with_pipeline_without_model(no_model_pipeline):
    model_step = Base().get_model_step_from_pipeline(no_model_pipeline)
    assert model_step is None, "Should return None for a pipeline without a model step."


def test_run_name_extraction_with_valid_pipeline(valid_pipeline):
    run_name = Base().get_run_name(valid_pipeline)
    assert run_name == 'pca|rf', "Failed to generate correct run name for a valid pipeline."


def test_ada_boost_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('AdaBoostRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_ada_boost_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('AdaBoostRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('AdaBoostRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_ard_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('ARDRegression', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_ard_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('ARDRegression', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('ARDRegression', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_bayesian_ridge_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('BayesianRidge', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_bayesian_ridge_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('BayesianRidge', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('BayesianRidge', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_decision_tree_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('DecisionTreeRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_decision_tree_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('DecisionTreeRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('DecisionTreeRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_elastic_net_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('ElasticNet', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_elastic_net_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('ElasticNet', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('ElasticNet', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_extra_tree_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('ExtraTreeRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_extra_tree_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('ExtraTreeRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('ExtraTreeRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_extra_trees_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('ExtraTreesRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_extra_trees_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('ExtraTreesRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('ExtraTreesRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_gamma_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('GammaRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_gamma_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('GammaRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('GammaRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_gaussian_process_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('GaussianProcessRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_gaussian_process_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('GaussianProcessRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('GaussianProcessRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_gradient_boosting_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('GradientBoostingRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_gradient_boosting_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('GradientBoostingRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('GradientBoostingRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_hist_gradient_boosting_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('HistGradientBoostingRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_hist_gradient_boosting_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('HistGradientBoostingRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('HistGradientBoostingRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_isotonic_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('IsotonicRegression', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_isotonic_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('IsotonicRegression', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('IsotonicRegression', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_kneighbours_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('KNeighborsRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_kneighbours_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('KNeighborsRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('KNeighborsRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_hubber_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('HuberRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_hubber_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('HuberRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('HuberRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_lars_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('Lars', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_lars_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('Lars', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('Lars', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_lasso_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('Lasso', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_lasso_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('Lasso', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('Lasso', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_lasso_lars_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('LassoLars', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_lasso_lars_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LassoLars', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LassoLars', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_lasso_lars_ic_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('LassoLarsIC', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_lasso_lars_ic_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LassoLarsIC', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LassoLarsIC', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_lgbm_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('LGBMRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_lgbm_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LGBMRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LGBMRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_linear_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('LinearRegression', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_linear_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LinearRegression', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LinearRegression', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_linear_svr_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('LinearSVR', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_linear_svr_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LinearSVR', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LinearSVR', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_mlp_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('MLPRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_mlp_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('MLPRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('MLPRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_nu_svr_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('NuSVR', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_nu_svr_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('NuSVR', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('NuSVR', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_orthogonal_matching_pursuit_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('OrthogonalMatchingPursuit', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_orthogonal_matching_pursuit_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('OrthogonalMatchingPursuit', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('OrthogonalMatchingPursuit', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_passive_aggressive_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('PassiveAggressiveRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_passive_aggressive_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('PassiveAggressiveRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('PassiveAggressiveRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_poisson_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('PoissonRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_poisson_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('PoissonRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('PoissonRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_rf_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('RandomForestRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_rf_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('RandomForestRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('RandomForestRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_ridge_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('Ridge', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_ridge_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('Ridge', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('Ridge', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_sgr_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('SGDRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_sgr_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('SGDRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('SGDRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_svr_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('SVR', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_svr_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('SVR', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('SVR', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_theil_sen_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('TheilSenRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_theil_sen_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('TheilSenRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('TheilSenRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_tweedie_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('TweedieRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_tweedie_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('TweedieRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('TweedieRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_xgboost_regressor_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('XGBRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_xgboost_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('XGBRegressor', REGRESSOR_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('XGBRegressor', REGRESSOR_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=df.drop(columns=['target']),
                      y=df['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_ada_boost_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('AdaBoostClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_ada_boost_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('AdaBoostClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('AdaBoostClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_decision_tree_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('DecisionTreeClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_decision_tree_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('DecisionTreeClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('DecisionTreeClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_extra_trees_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('ExtraTreesClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_extra_trees_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('ExtraTreesClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('ExtraTreesClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_gaussian_process_classifier():
    # Check that the function returns a dictionary
    Base().get_basic_pipeline('GaussianProcessClassifier', CLASSIFIER_KEY)


def test_gradient_boosting_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('GradientBoostingClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_gradient_boosting_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('GradientBoostingClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('GradientBoostingClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_knn_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('KNeighborsClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_knn_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('KNeighborsClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('KNeighborsClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_lgbm_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('LGBMClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_lgbm_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LGBMClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LGBMClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_label_propagation_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('LabelPropagation', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_label_propagation_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LabelPropagation', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LabelPropagation', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_label_spreading_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('LabelSpreading', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_label_spreading_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LabelSpreading', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LabelSpreading', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_linear_svc_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('LinearSVC', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_linear_svc_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LinearSVC', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LinearSVC', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_linear_discriminant_analysis_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('LinearDiscriminantAnalysis', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_linear_discriminant_analysis_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LinearDiscriminantAnalysis', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LinearDiscriminantAnalysis', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_logistic_regression_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('LogisticRegression', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_logistic_regression_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LogisticRegression', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LogisticRegression', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_mlp_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('MLPClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_mlp_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('MLPClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('MLPClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_nearest_centroid_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('NearestCentroid', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_nearest_centroid_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('NearestCentroid', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('NearestCentroid', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_nu_svc_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('NuSVC', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_nu_svc_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('NuSVC', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('NuSVC', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_passive_aggressive_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('PassiveAggressiveClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_passive_aggressive_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('PassiveAggressiveClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('PassiveAggressiveClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_perceptron_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('Perceptron', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_perceptron_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('Perceptron', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('Perceptron', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_qda_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('QuadraticDiscriminantAnalysis', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_qda_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('QuadraticDiscriminantAnalysis', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('QuadraticDiscriminantAnalysis', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_random_forest_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('RandomForestClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_random_forest_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('RandomForestClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('RandomForestClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_ridge_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('RidgeClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_ridge_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('RidgeClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('RidgeClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_sgd_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('SGDClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_sgd_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('SGDClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('SGDClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_svc_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('SVC', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_svc_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('SVC', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('SVC', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."


def test_xgboost_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('XGBClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_xgboost_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('XGBClassifier', CLASSIFIER_KEY)
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('XGBClassifier', CLASSIFIER_KEY)
    model_step = Base().get_model_step_from_pipeline(pipeline)
    param_dist = {f'{model_step}__{i}': j for i, j in param_dist.items()}

    # Initialize the RandomizedSearchCV with RandomForestRegressor
    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=5)

    # Run RandomizedSearchCV on the dataset
    random_search.fit(X=iris.drop(columns=['target']),
                      y=iris['target'])

    # Check if RandomizedSearchCV runs successfully
    assert hasattr(random_search, 'best_estimator_'), "RandomizedSearchCV should have an attribute 'best_estimator_' after fitting."
