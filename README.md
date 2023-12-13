Example run:

typical approach to loading in data and defining an sklearn pipeline
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

**Not sure which model to run?** - No problem
`mlflowgo` can quickly identify and "battle" the top contenders to find the best solution. Consider the below code:

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