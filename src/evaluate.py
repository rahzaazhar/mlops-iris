import typer
from typing_extensions import Annotated
from src.data import load_data, preprocess
from src import predict
from src.predict import SklearnPredictor, get_mlflow_checkpoint
from src import utils
from sklearn.metrics import precision_recall_fscore_support

app = typer.Typer()

@app.command()
def evaluate(run_id : Annotated[str, typer.Option(help='name of the experiment for this training workload')],
             results_fp : Annotated[str, typer.Option(help='file path to store evaluation results')],
             ):
    
    data = load_data()
    ds = data['data']
    ds = preprocess(ds)

    checkpoint = get_mlflow_checkpoint(run_id)
    predictor = SklearnPredictor.from_checkpoint(checkpoint)
    y_pred = predictor(ds['X'])
    metrics = precision_recall_fscore_support(ds['targets'], y_pred, average="weighted")
    performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}

    if results_fp:  # pragma: no cover, saving results
        utils.save_dict(d=performance, path=results_fp)

    return performance
    
if __name__ == "__main__":
    app()