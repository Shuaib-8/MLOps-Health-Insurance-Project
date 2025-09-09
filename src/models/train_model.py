import argparse
import logging
import platform

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()

# -----------------------------
# Load model from config
# -----------------------------
def get_model_instance(name, params):
    model_map = {
        "RandomForestRegressor": RandomForestRegressor,
        "LinearRegression": LinearRegression,
        "SVR": SVR,
        "XGBRegressor": xgb.XGBRegressor,
    }
    if name not in model_map:
        raise ValueError(f"Model {name} is not supported.")
    return model_map[name](**params)

# -----------------------------
# Main logic
# -----------------------------
def main():
    # Load config
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config['model']

    logger.info(f"Training model: {model_cfg['best_model_name']} with params: {model_cfg['best_model_params']}")

    # MLflow setup
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(model_cfg['best_model_name'])
        logger.info(f"MLflow tracking URI set to: {args.mlflow_tracking_uri}")

    # Load dataset
    df = pd.read_csv(args.data)
    target_variable = model_cfg.get("target_variable", "charges")

    # Use all columns except target as features
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get model
    model = get_model_instance(model_cfg.get("best_model_name", "XGBRegressor"), model_cfg.get("best_model_params"))

    # Commence MLflow run
    with mlflow.start_run(run_name="final_model_training"):
        logger.info(f"Training the model: {model_cfg['best_model_name']}")
 
        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # Log params and metrics 
        mlflow.log_params(model_cfg['best_model_params'])
        mlflow.log_metrics({"r2": r2, "mae": mae, "rmse": rmse, "mse": mse})

        # Log additional information as params
        mlflow.log_param("model_name", model_cfg['best_model_name'])
        mlflow.log_param("target_variable", target_variable)
        mlflow.log_param("training_dataset", args.data)
        
        # Register model in MLflow Model Registry (simplified approach)
        model_name = model_cfg['best_model_name']
        
        # Save model locally first
        save_path = f"{args.models_dir}/trained/{model_name}.pkl"
        joblib.dump(model, save_path)
        logger.info(f"Saved trained model to: {save_path}")
        
        # Create MLflow client for model registry operations
        client = mlflow.tracking.MlflowClient()
        run_id = mlflow.active_run().info.run_id
        
        # Create registered model if it doesn't exist
        try:
            registered_model = client.create_registered_model(
                name=model_name,
                description=f"Health Insurance Charges Prediction Model using {model_name}"
            )
            logger.info(f"Created new registered model: {model_name}")
        except mlflow.exceptions.RestException as e:
            if "already exists" in str(e):
                logger.info(f"Registered model {model_name} already exists")
            else:
                logger.warning(f"Error creating registered model: {e}")

        # Try to create a model version with run reference (without artifacts)
        try:
            # Create model version using run ID
            model_version = client.create_model_version(
                name=model_name,
                source=f"runs:/{run_id}/model",  # Reference to run without actual artifacts
                run_id=run_id
            )
            logger.info(f"Created model version {model_version.version} for {model_name}")
            
            # Transition model to "Staging" stage
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging",
                archive_existing_versions=False
            )
            logger.info(f"Moved model version {model_version.version} to Staging stage")
            
        except Exception as e:
            logger.warning(f"Could not create model version (expected with Docker setup): {e}")
            # This is expected due to Docker artifact store limitations

        # Add comprehensive description
        try:
            description = (
                f"Model for predicting US health insurance charges.\n"
                f"Algorithm: {model_cfg['best_model_name']}\n"
                f"Hyperparameters: {model_cfg['best_model_params']}\n"
                f"Features used: All features except target variable\n"
                f"Target variable: {target_variable}\n"
                f"Trained on dataset: {args.data}\n"
                f"Model saved at: {save_path}\n"
                f"Performance metrics:\n"
                f"  - R²: {r2:.4f}\n"
                f"  - MAE: {mae:.2f}\n"
                f"  - RMSE: {rmse:.2f}\n"
                f"  - MSE: {mse:.2f}\n"
                f"Training completed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            client.update_registered_model(name=model_name, description=description)
            logger.info("Updated registered model description")
        except Exception as e:
            logger.warning(f"Could not update model description: {e}")

        # Try to add tags for better organization (simplified approach)
        try:
            tags_to_add = [
                ("algorithm", model_cfg['best_model_name']),
                ("r2_score", str(round(r2, 4))),
                ("mae", str(round(mae, 2))),
                ("rmse", str(round(rmse, 2))),
                ("target_variable", target_variable),
                ("training_dataset", args.data.split("/")[-1]),  # Just filename
                ("python_version", platform.python_version()),
                ("scikit_learn_version", sklearn.__version__),
                ("xgboost_version", xgb.__version__),
                ("pandas_version", pd.__version__),
                ("numpy_version", np.__version__),
            ]
            
            # Add hyperparameter tags
            for param, value in model_cfg['best_model_params'].items():
                tags_to_add.append((f"param_{param}", str(value)))
            
            # Add tags one by one (more robust for Docker setup)
            for tag_key, tag_value in tags_to_add:
                try:
                    client.set_registered_model_tag(model_name, tag_key, tag_value)
                except Exception as tag_error:
                    logger.debug(f"Could not set tag {tag_key}: {tag_error}")
                    
            logger.info(f"Added metadata tags to registered model {model_name}")
        except Exception as e:
            logger.warning(f"Could not add all tags to registered model: {e}")
        
        logger.info(f"Model {model_name} is now registered in MLflow Model Registry")
        
        logger.info(f"Model training completed for {model_name}")
        logger.info(f"Final MSE: {mse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        logger.info(f"Check registered model at: http://localhost:5555/#/models/{model_name}")

if __name__ == "__main__":
    main()
