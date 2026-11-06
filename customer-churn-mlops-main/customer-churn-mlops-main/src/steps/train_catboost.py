import pandas as pd
import optuna
import mlflow
import mlflow.catboost

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from zenml import step


@step
def train_catboost_model(train_df: pd.DataFrame):
    """Train CatBoost model using Optuna and log the final model to MLflow."""

    print("Starting CatBoost hyperparameter tuning...")

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    mlflow.set_experiment("customer_churn_catboost")

    def objective(trial):

        params = {
            "iterations": trial.suggest_int(
                "iterations", 100, 500
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "depth": trial.suggest_int(
                "depth", 4, 10
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 10.0
            ),
        }

        model = CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_train).astype(int).ravel()
        probabilities = model.predict_proba(X_train)[:, 1]

        accuracy = accuracy_score(
            y_train,
            predictions,
        )

        f1 = f1_score(
            y_train,
            predictions,
        )

        roc_auc = roc_auc_score(
            y_train,
            probabilities,
        )

        with mlflow.start_run(
            run_name=f"catboost_optuna_trial_{trial.number}"
        ):
            mlflow.log_params(params)

            mlflow.log_metric(
                "accuracy",
                accuracy,
            )

            mlflow.log_metric(
                "f1_score",
                f1,
            )

            mlflow.log_metric(
                "roc_auc",
                roc_auc,
            )

        return roc_auc

    study = optuna.create_study(
        direction="maximize",
        study_name="customer_churn_catboost",
    )

    study.optimize(
        objective,
        n_trials=35,
    )

    print("\nCatBoost tuning completed.")

    print(
        f"Number of trials: {len(study.trials)}"
    )

    print(
        f"Best ROC-AUC: {study.best_value:.4f}"
    )

    print("Best parameters:")

    for parameter, value in study.best_params.items():
        print(f"  {parameter}: {value}")

    # Train final CatBoost model using the best parameters.
    best_model = CatBoostClassifier(
        **study.best_params,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    best_model.fit(
        X_train,
        y_train,
    )

    predictions = best_model.predict(
        X_train
    ).astype(int).ravel()

    probabilities = best_model.predict_proba(
        X_train
    )[:, 1]

    accuracy = accuracy_score(
        y_train,
        predictions,
    )

    f1 = f1_score(
        y_train,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_train,
        probabilities,
    )

    # Log and register the final CatBoost model.
    with mlflow.start_run(
        run_name="best_catboost_model"
    ):

        mlflow.log_params(
            study.best_params
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc,
        )

        mlflow.log_param(
            "model_type",
            "CatBoostClassifier",
        )

        mlflow.set_tag(
            "registered_model",
            "CustomerChurnGradientBoosting",
        )

        mlflow.catboost.log_model(
            best_model,
            name="catboost_model",
            registered_model_name="CustomerChurnGradientBoosting",
        )

    print("\nFinal CatBoost model trained successfully.")

    print(f"Training Accuracy: {accuracy:.4f}")
    print(f"Training F1 Score: {f1:.4f}")
    print(f"Training ROC-AUC: {roc_auc:.4f}")

    return best_model
