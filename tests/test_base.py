import pytest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, make_friedman1
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
from mlflowgo.base import Base


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
    param_dist = Base().get_param_dist('AdaBoostRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_ada_boost_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('AdaBoostRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('AdaBoostRegressor')
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
    param_dist = Base().get_param_dist('ARDRegression')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_ard_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('ARDRegression')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('ARDRegression')
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
    param_dist = Base().get_param_dist('BayesianRidge')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_bayesian_ridge_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('BayesianRidge')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('BayesianRidge')
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
    param_dist = Base().get_param_dist('DecisionTreeRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_decision_tree_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('DecisionTreeRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('DecisionTreeRegressor')
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
    param_dist = Base().get_param_dist('ElasticNet')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_elastic_net_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('ElasticNet')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('ElasticNet')
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
    param_dist = Base().get_param_dist('ExtraTreesRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_extra_trees_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('ExtraTreesRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('ExtraTreesRegressor')
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
    param_dist = Base().get_param_dist('GradientBoostingRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_gradient_boosting_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('GradientBoostingRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('GradientBoostingRegressor')
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
    param_dist = Base().get_param_dist('KNeighborsRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_kneighbours_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('KNeighborsRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('KNeighborsRegressor')
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
    param_dist = Base().get_param_dist('HuberRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_hubber_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('HuberRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('HuberRegressor')
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
    param_dist = Base().get_param_dist('Lars')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_lars_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('Lars')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('Lars')
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
    param_dist = Base().get_param_dist('Lasso')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_lasso_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('Lasso')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('Lasso')
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
    param_dist = Base().get_param_dist('LassoLars')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_lasso_lars_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LassoLars')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LassoLars')
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
    param_dist = Base().get_param_dist('LGBMRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_lgbm_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LGBMRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LGBMRegressor')
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
    param_dist = Base().get_param_dist('LinearRegression')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_linear_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LinearRegression')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LinearRegression')
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
    param_dist = Base().get_param_dist('MLPRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_mlp_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('MLPRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('MLPRegressor')
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
    param_dist = Base().get_param_dist('OrthogonalMatchingPursuit')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_orthogonal_matching_pursuit_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('OrthogonalMatchingPursuit')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('OrthogonalMatchingPursuit')
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
    param_dist = Base().get_param_dist('PassiveAggressiveRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_passive_aggressive_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('PassiveAggressiveRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('PassiveAggressiveRegressor')
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
    param_dist = Base().get_param_dist('RandomForestRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_rf_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('RandomForestRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('RandomForestRegressor')
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
    param_dist = Base().get_param_dist('Ridge')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_ridge_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('Ridge')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('Ridge')
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
    param_dist = Base().get_param_dist('SGDRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_sgr_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('SGDRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('SGDRegressor')
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
    param_dist = Base().get_param_dist('SVR')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_svr_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('SVR')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('SVR')
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
    param_dist = Base().get_param_dist('TheilSenRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_theil_sen_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('TheilSenRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('TheilSenRegressor')
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
    param_dist = Base().get_param_dist('XGBRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_xgboost_regressor_randomized_search(df):
    # Get parameter distribution
    param_dist = Base().get_param_dist('XGBRegressor')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('XGBRegressor')
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
    param_dist = Base().get_param_dist('AdaBoostClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_ada_boost_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('AdaBoostClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('AdaBoostClassifier')
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
    param_dist = Base().get_param_dist('DecisionTreeClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_decision_tree_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('DecisionTreeClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('DecisionTreeClassifier')
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
    param_dist = Base().get_param_dist('ExtraTreesClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_extra_trees_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('ExtraTreesClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('ExtraTreesClassifier')
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
    Base().get_basic_pipeline('GaussianProcessClassifier')


def test_gradient_boosting_classifier_param_dist():
    # Check that the function returns a dictionary
    param_dist = Base().get_param_dist('GradientBoostingClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_gradient_boosting_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('GradientBoostingClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('GradientBoostingClassifier')
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
    param_dist = Base().get_param_dist('KNeighborsClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_knn_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('KNeighborsClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('KNeighborsClassifier')
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
    param_dist = Base().get_param_dist('LGBMClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_lgbm_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LGBMClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LGBMClassifier')
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
    param_dist = Base().get_param_dist('LabelPropagation')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_label_propagation_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LabelPropagation')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LabelPropagation')
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
    param_dist = Base().get_param_dist('LabelSpreading')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_label_spreading_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LabelSpreading')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LabelSpreading')
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
    param_dist = Base().get_param_dist('LinearSVC')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_linear_svc_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LinearSVC')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LinearSVC')
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
    param_dist = Base().get_param_dist('LinearDiscriminantAnalysis')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_linear_discriminant_analysis_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LinearDiscriminantAnalysis')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LinearDiscriminantAnalysis')
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
    param_dist = Base().get_param_dist('LogisticRegression')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_logistic_regression_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('LogisticRegression')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('LogisticRegression')
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
    param_dist = Base().get_param_dist('MLPClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_mlp_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('MLPClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('MLPClassifier')
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
    param_dist = Base().get_param_dist('Perceptron')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_perceptron_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('Perceptron')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('Perceptron')
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
    param_dist = Base().get_param_dist('QuadraticDiscriminantAnalysis')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_qda_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('QuadraticDiscriminantAnalysis')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('QuadraticDiscriminantAnalysis')
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
    param_dist = Base().get_param_dist('RandomForestClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_random_forest_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('RandomForestClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('RandomForestClassifier')
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
    param_dist = Base().get_param_dist('RidgeClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_ridge_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('RidgeClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('RidgeClassifier')
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
    param_dist = Base().get_param_dist('SGDClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_sgd_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('SGDClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('SGDClassifier')
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
    param_dist = Base().get_param_dist('SVC')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_svc_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('SVC')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('SVC')
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
    param_dist = Base().get_param_dist('XGBClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."


def test_xgboost_classifier_randomized_search(iris):
    # Get parameter distribution
    param_dist = Base().get_param_dist('XGBClassifier')
    assert isinstance(param_dist, dict), "Should return a dictionary."

    pipeline = Base().get_basic_pipeline('XGBClassifier')
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
