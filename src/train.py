import mlflow
import typer
from typing_extensions import Annotated
from src.data import preprocess, load_data
from src.config import MLFLOW_TRACKING_URI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
# Initialize Typer CLI app
app = typer.Typer()

@app.command()
def train_model(
    experiment_name: Annotated[str, typer.Option(help='name of the experiment for this training workload')] = None,
    n_estimators: Annotated[int, typer.Option(help='number of estimators')] = 100,
    max_depth: Annotated[int, typer.Option(help='max depth of trees')] = None,
    min_samples_split: Annotated[int, typer.Option(help='minimum number of samples needed for split')] = 2,
    ):
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    data = load_data()
    ds = data['data']

    mlflow.end_run() # figure out this weird bug 
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)

        train_df, val_df = train_test_split(ds, stratify=ds.target, test_size=0.2, random_state=1234)
        train_data = preprocess(train_df)
        val_data = preprocess(val_df)
        
        model = RandomForestClassifier(n_estimators=n_estimators, 
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split)
        model.fit(train_data['X'], train_data['targets'])
        y_preds = model.predict(val_data['X'])
        metrics = precision_recall_fscore_support(val_data['targets'], y_preds, average="weighted")
        mlflow.log_metric('precision', metrics[0])
        mlflow.log_metric('recall', metrics[1])
        mlflow.log_metric('f1', metrics[2])

        mlflow.sklearn.log_model(model, 'model')
        
    

if __name__ == "__main__":
    app()
