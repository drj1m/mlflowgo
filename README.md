
**mlflowgo** is a Python package that simplifies and enhances the integration of MLflow with your machine learning workflows. It provides a set of utilities and helper functions to streamline the process of tracking experiments, logging artifacts, and managing machine learning models using MLflow. 

Spend your time on evaluating, refining and productionising your models rather than recycling out the same code - grab a tea/coffee and let mlflowgo do the rest.

## Features

- Quickly find an appropriate ML model for your dataset.
- Log various types of artifacts, such as plots, reports, and data samples.
- Streamline the tracking of model parameters and metrics.
- Supports a range of machine learning frameworks and libraries.

## Installation

You can install **mlflowgo** using pip:

```bash
pip install mlflowgo
```

When using mlflowgo, if you receive an error relating to the missing file: `libomp.dylib`, try running `conda install -c conda-forge lightgbm`

## Example run 1:

Simple example using the sklearn dataset, in this case we assume some prior knowledge of which model will work best for this dataset.

``` python
from mlflowgo.mlflowgo import MLFlowGo
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris()
data = pd.DataFrame(
    data=np.c_[iris['data'],
               iris['target']],
    columns=np.append(iris['feature_names'], ['target'])
)

model = Pipeline(
        [('model', RandomForestClassifier(n_estimators=100))]
    )
```

We can leverage mlflowgo with:
``` python
mlflow_go = MLFlowGo(experiment_name="Iris_Classification_Experiment")

mlflow_go.run_experiment(pipeline=model,
                         X=data.drop(columns=['target']),
                         y=data['target'])
```

## Example run 2

Using the same Iris toy dataset we can run an experiment with mlflowgo without defining any pipelines up front.

``` python
iris = datasets.load_iris()
data = pd.DataFrame(
    data=np.c_[iris['data'],
               iris['target']],
    columns=np.append(iris['feature_names'], ['target'])
)

mlflow_go = MLFlowGo(experiment_name="Iris_Classification_Experiment")

mlflow_go.run_experiment(X=data.drop(columns=['target']),
                         y=data['target'])
```