"""Короткий EDA: баланс классов, срезы дефолта и связи с платежной историей"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from credit_default.config import project_root
from credit_default.data.features import (
    TARGET,
    add_features,
    normalize_demographics,
    standardize_columns,
)
from credit_default.data.load import ensure_raw_data

EDUCATION_LABELS = {
    0: "неизвестно",
    1: "аспирантура",
    2: "университет",
    3: "школа",
    4: "другое",
    5: "неизвестно",
    6: "неизвестно",
}
SEX_LABELS = {1: "мужчина", 2: "женщина"}
MARRIAGE_LABELS = {0: "другое", 1: "женат/замужем", 2: "не в браке", 3: "другое"}


def _bar_rate(frame: pd.DataFrame, column: str, ax, title: str) -> None:
    rates = frame.groupby(column, observed=True)[TARGET].mean().sort_index()
    rates.plot(kind="bar", ax=ax, color="#1d4e89")
    ax.set_title(title)
    ax.set_ylabel("Доля дефолта")
    ax.set_xlabel("")
    ax.set_ylim(0, max(0.5, float(rates.max()) + 0.05))
    ax.tick_params(axis="x", rotation=30)


def run_eda(destination: Path | None = None) -> dict:
    root = project_root()
    destination = destination or (root / "reports" / "figures")
    destination.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ensure_raw_data())
    standardized = standardize_columns(raw)
    featured = add_features(normalize_demographics(standardized))

    summary = {
        "rows": int(len(featured)),
        "columns": int(featured.shape[1]),
        "nulls": int(standardized.isna().sum().sum()),
        "default_rate": float(featured[TARGET].mean()),
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    counts = featured[TARGET].value_counts().sort_index()
    ax.bar(["Нет дефолта", "Дефолт"], counts.values, color=["#7f8c8d", "#c0392b"])
    ax.set_title("Баланс целевой переменной")
    ax.set_ylabel("Клиенты")
    fig.tight_layout()
    fig.savefig(destination / "eda_target_balance.png", dpi=120)
    plt.close(fig)

    plot_frame = featured.copy()
    plot_frame["education_label"] = plot_frame["EDUCATION"].map(EDUCATION_LABELS)
    plot_frame["sex_label"] = plot_frame["SEX"].map(SEX_LABELS)
    plot_frame["marriage_label"] = plot_frame["MARRIAGE"].map(MARRIAGE_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _bar_rate(plot_frame, "sex_label", axes[0], "Пол")
    _bar_rate(plot_frame, "education_label", axes[1], "Образование")
    _bar_rate(plot_frame, "marriage_label", axes[2], "Семейное положение")
    fig.tight_layout()
    fig.savefig(destination / "eda_default_by_demographics.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _bar_rate(featured, "age_bin", axes[0], "Бин возраста")
    late = featured.groupby("n_late_payments", observed=True)[TARGET].mean()
    late.plot(kind="bar", ax=axes[1], color="#c0392b")
    axes[1].set_title("Число просрочек за 6 месяцев")
    axes[1].set_ylabel("Доля дефолта")
    axes[1].set_xlabel("n_late_payments")
    fig.tight_layout()
    fig.savefig(destination / "eda_default_by_risk.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in ((0, "#7f8c8d"), (1, "#c0392b")):
        subset = featured.loc[featured[TARGET] == label, "LIMIT_BAL"]
        ax.hist(
            subset, bins=30, alpha=0.65, color=color, label="Дефолт" if label else "Нет"
        )
    ax.set_title("Кредитный лимит")
    ax.set_xlabel("LIMIT_BAL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "eda_limit_balance.png", dpi=120)
    plt.close(fig)

    corr_columns = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "utilization",
        "n_late_payments",
        "payment_ratio",
        TARGET,
    ]
    correlation = featured[corr_columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_columns)), corr_columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_columns)), corr_columns)
    for row_index, row_name in enumerate(corr_columns):
        for col_index, col_name in enumerate(corr_columns):
            ax.text(
                col_index,
                row_index,
                f"{correlation.loc[row_name, col_name]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title("Корреляции признаков с дефолтом")
    fig.tight_layout()
    fig.savefig(destination / "eda_correlation.png", dpi=120)
    plt.close(fig)

    summary_path = root / "reports" / "eda_summary.json"
    pd.Series(summary).to_json(summary_path, force_ascii=False)
    print(
        f"EDA: строк={summary['rows']}, пропусков в сырых полях={summary['nulls']}, "
        f"доля дефолта={summary['default_rate']:.3f}"
    )
    print(f"Графики: {destination}")
    return summary


def main() -> None:
    run_eda()


if __name__ == "__main__":
    main()
