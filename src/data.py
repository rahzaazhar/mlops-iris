import numpy as np
from sklearn.datasets import load_iris


def load_data():
    iris_bunch = load_iris(as_frame=True)
    iris_df = iris_bunch.frame
    return dict(data=iris_df)


def get_index_to_class_map():
    return np.array(["setosa", "versicolor", "virginica"], dtype="<U10")


def preprocess(data):
    # data = preprocess_step1(data)
    # data = preprocess_step2(data)

    return dict(X=data.drop(columns=["target"]).values, targets=data["target"].values)
