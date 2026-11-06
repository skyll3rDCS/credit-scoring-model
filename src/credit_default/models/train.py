"""Стадия train: несколько экспериментов, MLflow и выбор модели по CV ROC-AUC"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score

from credit_default.config import load_params, project_root
from credit_default.data.features import TARGET, model_matrix
from credit_default.models.evaluate import (
    classification_metrics,
    save_confusion_matrix,
    save_feature_importance,
    save_roc_curve,
)
from credit_default.models.pipeline import build_pipeline


def _xy(df: pd.DataFrame):
    features = model_matrix(df)
    target = df[TARGET].astype(int)
    return features, target


def _fit_cv(estimator, x_train, y_train, cv: int):
    scores = cross_val_score(
        estimator,
        x_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    estimator.fit(x_train, y_train)
    return estimator, float(scores.mean())


def _logreg(random_state: int, **kwargs) -> LogisticRegression:
    return LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        random_state=random_state,
        **kwargs,
    )


def _experiments(random_state: int, cv: int):
    """Пять с лишним прогонов: разные алгоритмы и сетка гиперпараметров"""
    baseline = build_pipeline(_logreg(random_state, C=1.0))
    balanced = build_pipeline(
        _logreg(random_state, C=0.5, class_weight="balanced"),
    )
    search = GridSearchCV(
        estimator=build_pipeline(_logreg(random_state)),
        param_grid={
            "model__C": [0.1, 1.0, 3.0],
            "model__class_weight": [None, "balanced"],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    forest = build_pipeline(
        RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
    )
    boosting = build_pipeline(
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.9,
            random_state=random_state,
        )
    )
    hist = build_pipeline(
        HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=150,
            class_weight="balanced",
            random_state=random_state,
        )
    )
    return (
        (
            "logreg_baseline",
            {"family": "logistic_regression", "C": 1.0},
            baseline,
            False,
        ),
        (
            "logreg_balanced",
            {"family": "logistic_regression", "C": 0.5, "class_weight": "balanced"},
            balanced,
            False,
        ),
        (
            "logreg_gridsearch",
            {"family": "logistic_regression", "search": "grid"},
            search,
            True,
        ),
        (
            "random_forest",
            {
                "family": "random_forest",
                "n_estimators": 120,
                "max_depth": 8,
                "class_weight": "balanced_subsample",
            },
            forest,
            False,
        ),
        (
            "gradient_boosting",
            {
                "family": "gradient_boosting",
                "n_estimators": 100,
                "learning_rate": 0.08,
                "max_depth": 3,
            },
            boosting,
            False,
        ),
        (
            "hist_gradient_boosting",
            {
                "family": "hist_gradient_boosting",
                "max_iter": 150,
                "learning_rate": 0.08,
                "max_depth": 6,
                "class_weight": "balanced",
            },
            hist,
            False,
        ),
    )


def _mlflow_params(params: dict) -> dict:
    safe = {}
    for key, value in params.items():
        if value is None:
            safe[key] = "None"
        elif isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def _log_model(model, x_train, y_train) -> None:
    signature = None
    try:
        from mlflow.models import infer_signature

        signature = infer_signature(
            x_train.head(50), model.predict_proba(x_train.head(50))[:, 1]
        )
    except Exception:
        signature = None

    log_model = mlflow.sklearn.log_model
    parameters = inspect.signature(log_model).parameters
    kwargs = {
        "sk_model": model,
        "serialization_format": "cloudpickle",
    }
    if "name" in parameters:
        kwargs["name"] = "model"
    else:
        kwargs["artifact_path"] = "model"
    if signature is not None and "signature" in parameters:
        kwargs["signature"] = signature
    if "sk_model" not in parameters:
        kwargs.pop("sk_model")
        log_model(model, **kwargs)
        return
    log_model(**kwargs)


def _fit_one(name, logged_params, estimator, is_search, x_train, y_train, cv: int):
    if is_search:
        estimator.fit(x_train, y_train)
        model = estimator.best_estimator_
        cv_auc = float(estimator.best_score_)
        best_params = {key: value for key, value in estimator.best_params_.items()}
        logged_params = {**logged_params, **best_params, "cv_folds": cv}
        grid_frame = pd.DataFrame(estimator.cv_results_)
        keep = [
            column
            for column in grid_frame.columns
            if column.startswith("param_")
            or column in {"mean_test_score", "rank_test_score"}
        ]
        return model, logged_params, cv_auc, grid_frame[keep]
    model, cv_auc = _fit_cv(estimator, x_train, y_train, cv)
    logged_params = {**logged_params, "cv_folds": cv}
    return model, logged_params, cv_auc, None


def _configure_tracking(root: Path, experiment_name: str) -> None:
    """SQLite-трекинг: файловый backend в MLflow 3 отключен без отдельного флага"""
    database = (root / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{database}")
    artifact_root = (root / "mlartifacts").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name,
            artifact_location=artifact_root.as_uri(),
        )
    mlflow.set_experiment(experiment_name)


def run_training() -> dict:
    params = load_params()
    train_params = params["train"]
    root = project_root()
    train_df = pd.read_csv(root / "data" / "processed" / "train.csv")
    test_df = pd.read_csv(root / "data" / "processed" / "test.csv")
    x_train, y_train = _xy(train_df)
    x_test, y_test = _xy(test_df)
    threshold = float(train_params["threshold"])
    cv = int(train_params["cv_folds"])
    random_state = int(train_params["random_state"])

    _configure_tracking(root, train_params["experiment_name"])

    figures = root / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    records = []
    best = None

    for name, logged_params, estimator, is_search in _experiments(random_state, cv):
        model, run_params, cv_auc, grid = _fit_one(
            name,
            logged_params,
            estimator,
            is_search,
            x_train,
            y_train,
            cv,
        )
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = classification_metrics(y_test, probabilities, threshold=threshold)
        metrics["cv_roc_auc"] = cv_auc
        roc_path = figures / f"roc_{name}.png"
        save_roc_curve(
            y_test,
            probabilities,
            roc_path,
            title=f"ROC-кривая: {name}",
        )
        with mlflow.start_run(run_name=name):
            mlflow.log_params(_mlflow_params(run_params))
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(roc_path))
            if grid is not None:
                grid_path = root / "reports" / "logreg_grid.csv"
                grid.to_csv(grid_path, index=False)
                mlflow.log_artifact(str(grid_path))
            _log_model(model, x_train, y_train)

        record = {"name": name, **metrics, "params": run_params}
        records.append(record)
        print(
            f"{name}: cv_auc={cv_auc:.4f} test_auc={metrics['roc_auc']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )
        if best is None or cv_auc > best["cv_roc_auc"]:
            best = {
                "name": name,
                "model": model,
                "probabilities": probabilities,
                "metrics": metrics,
                "cv_roc_auc": cv_auc,
                "params": run_params,
            }

    assert best is not None
    predictions = (best["probabilities"] >= threshold).astype(int)
    save_roc_curve(
        y_test,
        best["probabilities"],
        figures / "roc_curve.png",
        title=f"ROC-кривая выбранной модели ({best['name']})",
    )
    save_confusion_matrix(
        y_test,
        predictions,
        figures / "confusion_matrix.png",
        title=f"Матрица ошибок: {best['name']}",
    )
    sample = x_test.sample(n=min(1500, len(x_test)), random_state=random_state)
    sample_target = y_test.loc[sample.index]
    importance = permutation_importance(
        best["model"],
        sample,
        sample_target,
        n_repeats=4,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )
    save_feature_importance(
        list(sample.columns),
        importance.importances_mean,
        figures / "feature_importance.png",
        title=f"Permutation importance: {best['name']}",
    )

    model_path = root / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], model_path)

    comparison = pd.DataFrame(
        [
            {
                "name": item["name"],
                "cv_roc_auc": item["cv_roc_auc"],
                "roc_auc": item["roc_auc"],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1"],
            }
            for item in records
        ]
    ).sort_values("cv_roc_auc", ascending=False)
    reports = root / "reports"
    comparison.to_csv(reports / "experiments.csv", index=False)

    payload = {
        "selected_model": best["name"],
        "selection_metric": "cv_roc_auc",
        "threshold": threshold,
        "test_size": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
        "cv_roc_auc": best["cv_roc_auc"],
        "test_metrics": {
            key: best["metrics"][key]
            for key in ("roc_auc", "precision", "recall", "f1")
        },
        "params": {key: _jsonable(value) for key, value in best["params"].items()},
        "experiments": comparison.to_dict(orient="records"),
        "model_path": str(model_path.relative_to(root)),
    }
    metrics_path = reports / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary(reports / "EXPERIMENTS.md", payload)
    print(f"Выбрана модель {best['name']} (CV ROC-AUC={best['cv_roc_auc']:.4f})")
    print(f"Артефакт: {model_path}")
    return payload


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_summary(path: Path, payload: dict) -> None:
    lines = [
        "# Эксперименты PD-модели",
        "",
        "Модель для сервиса выбрана по среднему ROC-AUC",
        "на кросс-валидации обучающей выборки.",
        "Метрики precision, recall и F1 ниже посчитаны на отложенном тесте при пороге "
        f"{payload['threshold']}.",
        "",
        f"Выбранный эксперимент: `{payload['selected_model']}`.",
        "",
        "| Эксперимент | CV ROC-AUC | Test ROC-AUC | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["experiments"]:
        lines.append(
            "| {name} | {cv_roc_auc:.4f} | {roc_auc:.4f} | {precision:.4f} | "
            "{recall:.4f} | {f1:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Доля дефолта в тесте: "
            f"{payload['positive_rate_test']:.3f}. "
            "Из-за дисбаланса классов порог 0.5 у моделей без балансировки весов "
            "обычно дает более высокую precision и более низкий recall. "
            "Для ранжирования заявок основной метрикой выбран ROC-AUC.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
