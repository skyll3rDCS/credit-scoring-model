"""Имитация потока новых заявок: скоринг через API и PSI относительно train"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

import joblib
import pandas as pd

from credit_default.api.service import decision_threshold, model_path, predict_frame
from credit_default.config import load_params, project_root
from credit_default.data.features import (
    RAW_FEATURES,
    TARGET,
    add_features,
    normalize_demographics,
)
from credit_default.monitoring.psi import population_stability_index, psi_band

MONITORED_FEATURES = (
    "LIMIT_BAL",
    "AGE",
    "utilization",
    "n_late_payments",
    "pay_status_mean",
)


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def score_via_api(
    frame: pd.DataFrame, api_url: str, batch_size: int = 200
) -> list[float]:
    records = json.loads(frame[list(RAW_FEATURES)].to_json(orient="records"))
    probabilities: list[float] = []
    endpoint = api_url.rstrip("/") + "/predict-batch"
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        try:
            response = _post_json(endpoint, {"records": chunk})
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"API недоступен: {endpoint}. Запустите make api. Причина: {exc}"
            ) from exc
        probabilities.extend(item["probability"] for item in response["predictions"])
    return probabilities


def _with_drift(raw: pd.DataFrame) -> pd.DataFrame:
    shifted = raw.copy()
    shifted["LIMIT_BAL"] = shifted["LIMIT_BAL"] * 2.5
    shifted["AGE"] = (shifted["AGE"] + 15).clip(upper=80)
    shifted["PAY_0"] = 2
    shifted["BILL_AMT1"] = shifted["LIMIT_BAL"] * 0.95
    return shifted


def _feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    return add_features(normalize_demographics(raw))


def run_simulation(
    api_url: str | None = None,
    sample_size: int | None = None,
    inject_drift: bool = False,
    skip_api: bool = False,
) -> dict:
    params = load_params()
    monitoring = params["monitoring"]
    api_url = api_url or monitoring["api_url"]
    sample_size = int(sample_size or monitoring["sample_size"])
    bins = int(monitoring["psi_bins"])
    warn = float(monitoring["psi_warn"])
    alert = float(monitoring["psi_alert"])
    root = project_root()

    train_df = pd.read_csv(root / "data" / "processed" / "train.csv")
    test_df = pd.read_csv(root / "data" / "processed" / "test.csv")
    sample = test_df.sample(
        n=min(sample_size, len(test_df)),
        random_state=int(params["prepare"]["random_state"]),
    )
    raw = sample[RAW_FEATURES].copy()
    if inject_drift:
        raw = _with_drift(raw)
    actual = _feature_frame(raw)

    reference_model = joblib.load(model_path())
    reference_scores = [
        item["probability"]
        for item in predict_frame(
            reference_model,
            train_df[RAW_FEATURES],
            threshold=decision_threshold(),
        )
    ]
    if skip_api:
        actual_scores = [
            item["probability"] for item in predict_frame(reference_model, raw)
        ]
        scored_by = "local_model"
    else:
        actual_scores = score_via_api(raw, api_url)
        scored_by = "api"

    feature_psi = {}
    for column in MONITORED_FEATURES:
        value = population_stability_index(
            train_df[column].to_numpy(),
            actual[column].to_numpy(),
            bins=bins,
        )
        feature_psi[column] = {"psi": value, "band": psi_band(value, warn, alert)}

    score_psi = population_stability_index(reference_scores, actual_scores, bins=bins)
    report = {
        "sample_size": int(len(raw)),
        "inject_drift": inject_drift,
        "scored_by": scored_by,
        "api_url": None if skip_api else api_url,
        "reference_rows": int(len(train_df)),
        "reference_default_rate": float(train_df[TARGET].mean()),
        "score_psi": {"psi": score_psi, "band": psi_band(score_psi, warn, alert)},
        "feature_psi": feature_psi,
        "mean_predicted_probability": float(sum(actual_scores) / len(actual_scores)),
    }
    destination = (
        root
        / "reports"
        / ("drift_report_shifted.json" if inject_drift else "drift_report.json")
    )
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_report(report, destination)
    return report


def _print_report(report: dict, destination) -> None:
    print(f"Отчет: {destination}")
    print(
        f"PSI вероятностей ({report['scored_by']}): "
        f"{report['score_psi']['psi']:.4f} [{report['score_psi']['band']}]"
    )
    for name, item in report["feature_psi"].items():
        print(f"PSI {name}: {item['psi']:.4f} [{item['band']}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="PSI новых заявок относительно train")
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--inject-drift", action="store_true")
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Считать вероятности локальной моделью, без HTTP",
    )
    args = parser.parse_args()
    run_simulation(
        api_url=args.api_url,
        sample_size=args.sample_size,
        inject_drift=args.inject_drift,
        skip_api=args.skip_api,
    )


if __name__ == "__main__":
    main()
