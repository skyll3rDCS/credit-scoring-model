"""Набор правил Great Expectations для сырого датасета кредитных карт"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from credit_default.data.features import (
    BILL_COLS,
    PAY_AMT_COLS,
    PAY_STATUS_COLS,
    REQUIRED_RAW_COLUMNS,
    TARGET,
    standardize_columns,
)


@dataclass
class ValidationReport:
    success: bool
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.success:
            return "Валидация пройдена"
        return "Валидация не пройдена: " + "; ".join(self.failed)


class DataValidationError(RuntimeError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(report.summary())


def _expectations():
    import great_expectations as gx

    expectations = [
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1, max_value=100_000),
    ]
    for column in REQUIRED_RAW_COLUMNS:
        expectations.append(gx.expectations.ExpectColumnToExist(column=column))
        expectations.append(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

    expectations.extend(
        [
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="LIMIT_BAL",
                min_value=1,
                max_value=1_500_000,
            ),
            gx.expectations.ExpectColumnValuesToBeInSet(column="SEX", value_set=[1, 2]),
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="EDUCATION",
                value_set=[0, 1, 2, 3, 4, 5, 6],
            ),
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="MARRIAGE",
                value_set=[0, 1, 2, 3],
            ),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="AGE",
                min_value=18,
                max_value=100,
            ),
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=TARGET,
                value_set=[0, 1],
            ),
        ]
    )
    for column in PAY_STATUS_COLS:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=-2,
                max_value=9,
            )
        )
    for column in BILL_COLS:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=-1_000_000,
                max_value=2_000_000,
            )
        )
    for column in PAY_AMT_COLS:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=0,
                max_value=2_000_000,
            )
        )
    return expectations


def _failed_messages(result) -> list[str]:
    rows = getattr(result, "results", None) or []
    messages: list[str] = []
    for row in rows:
        if getattr(row, "success", True):
            continue
        config = getattr(row, "expectation_config", None)
        expectation_type = getattr(config, "type", None) or "expectation"
        kwargs = getattr(config, "kwargs", {}) if config is not None else {}
        column = kwargs.get("column") if isinstance(kwargs, dict) else None
        if column:
            messages.append(f"{expectation_type}[{column}]")
        else:
            messages.append(str(expectation_type))
    if not messages and not getattr(result, "success", False):
        messages.append("suite failed")
    return messages


def validate_raw_dataframe(
    df: pd.DataFrame,
    already_standardized: bool = False,
) -> ValidationReport:
    """Проверяет сырой батч. Аномалии дают success=False"""
    import great_expectations as gx

    frame = df if already_standardized else standardize_columns(df)
    context = gx.get_context(mode="ephemeral")
    suite = context.suites.add(gx.ExpectationSuite(name="credit_card_raw_suite"))
    for expectation in _expectations():
        suite.add_expectation(expectation)

    data_source = context.data_sources.add_pandas(name="credit_pandas")
    asset = data_source.add_dataframe_asset(name="credit_clients")
    batch_definition = asset.add_batch_definition_whole_dataframe(
        name="whole_dataframe"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": frame})
    result = batch.validate(suite)
    return ValidationReport(
        success=bool(result.success), failed=_failed_messages(result)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Валидация CSV правилами Great Expectations"
    )
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    report = validate_raw_dataframe(frame)
    print(report.summary())
    if not report.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
