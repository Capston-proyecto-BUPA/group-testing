import  h2o
from h2o.automl import H2OAutoML


def prepare_data_train(train, val=None):
    x = train.columns
    y = x[-1]
    x.remove(y)
    train[y] = train[y]
    if val is not None:
        val[y] = val[y]
        return train, x, y, val
    return train, x, y

def train_models(train, val, experiment_name):
    
    train, x, y, val = prepare_data_train(train, val)

    models = H2OAutoML(max_runtime_secs=300,
                       seed=1,
                       exclude_algos=['DeepLearning'],
                       project_name=experiment_name,
                       nfolds=0,
                       sort_metric='MSE')

    models.train(x=x, y=y, training_frame=train, validation_frame=val, leaderboard_frame=val)

    return models
