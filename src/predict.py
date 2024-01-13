import mlflow
import pickle
import typer
import pandas as pd
from typing_extensions import Annotated
from pathlib import Path
from urllib.parse import urlparse
from src.data import preprocess, get_index_to_class_map

app = typer.Typer()

class SklearnPredictor:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, data):
        return self.model.predict(data)

    def predict_prob(self, data):
        return self.model.predict_proba(data)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        with open(checkpoint, 'rb') as file:
            model = pickle.load(file)
        return cls(model=model)

def format_prob(prob, index_to_class_map):
    d = {}
    for i, item in enumerate(prob):
        d[index_to_class_map[i]] = item
    return d

def predict_proba(ds, predictor):
    index_to_class_map = get_index_to_class_map()
    ds = preprocess(ds)
    y_prob = predictor.predict_prob(ds['X'])
    results = []
    for i, prob in enumerate(y_prob):
        tag = index_to_class_map[prob.argmax()]
        results.append({"prediction": tag, "probabilities": format_prob(prob, index_to_class_map)})
    return results

def get_mlflow_checkpoint(run_id):
    artifact_dir = urlparse(mlflow.get_run(run_id).info.artifact_uri).path
    checkpoint = str(Path(f"{artifact_dir}/model/model.pkl").absolute())
    return checkpoint

@app.command()
def predict(run_id: Annotated[str, typer.Option(help="id of the specific run to load from")] = None,
            sepal_length: Annotated[float, typer.Option(help="sepal length (cm)")] = None,
            sepal_width: Annotated[float, typer.Option(help="sepal width (cm)")] = None,
            petal_length: Annotated[float, typer.Option(help="petal length (cm)")] = None,
            petal_width: Annotated[float, typer.Option(help="petal length (cm)")] = None,
            ):

    checkpoint = get_mlflow_checkpoint(run_id)
    predictor = SklearnPredictor.from_checkpoint(checkpoint)

    sample_ds = pd.DataFrame({"sepal length (cm)": [sepal_length], "sepal width (cm)":[sepal_width], "petal length (cm)":[petal_length], "petal width (cm)":[petal_width], "target":[0]})
    results = predict_proba(sample_ds, predictor)
    return results

if __name__ == "__main__":
    app()
