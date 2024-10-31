from datetime import datetime

import mlflow
import pytz


def setup_mlflow_tracking(model_name: str):
    # Use model name as experiment name
    mlflow.set_experiment(model_name)

    # Generate timestamp for run name
    timestamp = datetime.now(tz=pytz.timezone("Europe/Paris")).strftime("%Y%m%d_%H%M%S")

    return mlflow.start_run(run_name=timestamp)
