from sklearn.metrics import mean_absolute_error, r2_score

def evaluate(y_true, y_preds, model_names):
    for y_pred, name in zip(y_preds, model_names):
        print(f"{name}: MAE={mean_absolute_error(y_true, y_pred):.2f}, R2={r2_score(y_true, y_pred):.2f}")

import mlflow

def log_experiment(model, X_train, X_test, y_train, y_test, model_name):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mlflow.log_param("model", model_name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, model_name)
