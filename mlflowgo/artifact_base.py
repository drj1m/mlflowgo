class ArtifactBase():

    def __init__(self, **kwargs) -> None:
        self.pipeline = kwargs.get('pipeline', None)
        self.X_train = kwargs.get('X_train', None)
        self.y_train = kwargs.get('y_train', None)
        self.X_test = kwargs.get('X_test', None)
        self.y_test = kwargs.get('y_test', None)
        self.model_step = kwargs.get('model_step', None)
        self.feature_names = kwargs.get('feature_names', None)
        self.metric = kwargs.get('metric', None)
        self.param_name = kwargs.get('param_name', None)
        self.param_range = kwargs.get('param_range', None)
        self.objective = kwargs.get('objective', None)
        self.dataset_desc = kwargs.get('dataset_desc', None)
