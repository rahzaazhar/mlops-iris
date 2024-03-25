import argparse

import pandas as pd
import uvicorn
from fastapi import FastAPI
from starlette.requests import Request

from src.predict import SklearnPredictor, get_mlflow_checkpoint, predict_proba

app = FastAPI()
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", help="run id to use for serving")
args = parser.parse_args()
checkpoint = get_mlflow_checkpoint(args.run_id)
predictor = SklearnPredictor.from_checkpoint(checkpoint)


@app.post("/predict/")
async def _predict(request: Request):
    data = await request.json()
    print("lol")
    sample_ds = pd.DataFrame(
        {
            "sepal length (cm)": [data.get("sepal_length")],
            "sepal width (cm)": [data.get("sepal_width")],
            "petal length (cm)": [data.get("petal_length")],
            "petal width (cm)": [data.get("petal_width")],
            "target": [0],
        }
    )

    results = predict_proba(sample_ds, predictor)

    return {"results": results}


if __name__ == "__main__":
    uvicorn.run(app)
