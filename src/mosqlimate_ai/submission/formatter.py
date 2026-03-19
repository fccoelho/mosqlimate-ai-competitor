"""Submission formatter for Mosqlimate Sprint 2025 competition.

Formats forecasts according to the Mosqlimate API specification
for dengue forecasting competition submissions.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class SubmissionFormatter:
    """Format forecasts for Mosqlimate API submission.

    Handles formatting of predictions according to the competition
    requirements, including prediction intervals at multiple levels.

    Args:
        model_id: Registered model ID from Mosqlimate platform
        predict_date: Date when prediction was made
        description: Prediction description
        commit: Git commit hash

    Example:
        >>> formatter = SubmissionFormatter(model_id=123)
        >>> formatter.format_state_forecast(forecast_df, "SP")
        >>> formatter.save_submissions("submissions/")
    """

    def __init__(
        self,
        model_id: Optional[int] = None,
        predict_date: Optional[str] = None,
        description: str = "",
        commit: Optional[str] = None,
    ):
        self.model_id = model_id
        self.predict_date = predict_date or date.today().isoformat()
        self.description = description
        self.commit = commit

        self.submissions: List[Dict[str, Any]] = []

    def format_state_forecast(
        self,
        forecast_df: pd.DataFrame,
        uf: Optional[str],
        adm_0: str = "BRA",
    ) -> Dict[str, Any]:
        """Format forecast for a single state.

        Args:
            forecast_df: DataFrame with columns:
                - date: Date of prediction
                - median: Median prediction
                - lower_50, upper_50: 50% prediction interval
                - lower_80, upper_80: 80% prediction interval
                - lower_90, upper_90: 90% prediction interval
                - lower_95, upper_95: 95% prediction interval
            uf: State abbreviation (e.g., "SP")
            adm_0: Country code (default: "BRA")

        Returns:
            Formatted submission dictionary
        """
        required_cols = ["date", "median"]
        for col in required_cols:
            if col not in forecast_df.columns:
                raise ValueError(f"Missing required column: {col}")

        interval_cols = [
            ("lower_50", "upper_50"),
            ("lower_80", "upper_80"),
            ("lower_90", "upper_90"),
            ("lower_95", "upper_95"),
        ]

        for lower, upper in interval_cols:
            if lower not in forecast_df.columns or upper not in forecast_df.columns:
                logger.warning(f"Missing interval columns: {lower}, {upper}")

        dates = forecast_df["date"].tolist()
        if isinstance(dates[0], str):
            dates = [d for d in dates]
        else:
            dates = [d.strftime("%Y-%m-%d") for d in dates]

        preds = forecast_df["median"].tolist()
        preds = [max(0, p) for p in preds]

        prediction = {
            "dates": dates,
            "preds": preds,
        }

        for lower, upper in interval_cols:
            if lower in forecast_df.columns and upper in forecast_df.columns:
                prediction[lower] = [max(0, v) for v in forecast_df[lower].tolist()]
                prediction[upper] = [max(0, v) for v in forecast_df[upper].tolist()]

        submission = {
            "model": self.model_id,
            "description": self.description or f"Forecast for {uf}",
            "commit": self.commit,
            "predict_date": self.predict_date,
            "adm_0": adm_0,
            "adm_1": uf,
            "adm_2": None,
            "adm_3": None,
            "prediction": prediction,
        }

        self.submissions.append(submission)
        return submission

    def format_national_forecast(
        self,
        forecast_df: pd.DataFrame,
        adm_0: str = "BRA",
    ) -> Dict[str, Any]:
        """Format national-level forecast.

        Args:
            forecast_df: DataFrame with national forecasts
            adm_0: Country code

        Returns:
            Formatted submission dictionary
        """
        submission = self.format_state_forecast(forecast_df, uf=None, adm_0=adm_0)
        submission["adm_1"] = None
        submission["description"] = self.description or "National forecast for Brazil"
        return submission

    def format_all_states(
        self,
        forecasts_by_state: Dict[str, pd.DataFrame],
        include_national: bool = False,
        national_forecast: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """Format forecasts for all states.

        Args:
            forecasts_by_state: Dictionary mapping state UF to forecast DataFrames
            include_national: Whether to include national forecast
            national_forecast: National-level forecast DataFrame

        Returns:
            List of formatted submissions
        """
        submissions = []

        for uf, forecast_df in forecasts_by_state.items():
            submission = self.format_state_forecast(forecast_df, uf)
            submissions.append(submission)
            logger.info(f"Formatted forecast for {uf}")

        if include_national and national_forecast is not None:
            submission = self.format_national_forecast(national_forecast)
            submissions.append(submission)
            logger.info("Formatted national forecast")

        self.submissions.extend(submissions)
        return submissions

    def save_submissions(
        self,
        output_dir: Union[str, Path],
        format: str = "json",
    ) -> List[Path]:
        """Save all submissions to files.

        Args:
            output_dir: Directory to save submissions
            format: Output format ('json' or 'csv')

        Returns:
            List of paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for submission in self.submissions:
            uf = submission.get("adm_1", "BR")
            filename = f"forecast_{uf}_{self.predict_date}"

            if format == "json":
                filepath = output_dir / f"{filename}.json"
                with open(filepath, "w") as f:
                    json.dump(submission, f, indent=2)
            elif format == "csv":
                filepath = output_dir / f"{filename}.csv"
                self._save_csv(submission, filepath)
            else:
                raise ValueError(f"Unknown format: {format}")

            saved_paths.append(filepath)
            logger.info(f"Saved submission: {filepath}")

        return saved_paths

    def _save_csv(self, submission: Dict[str, Any], filepath: Path) -> None:
        """Save submission as CSV.

        Args:
            submission: Submission dictionary
            filepath: Output path
        """
        prediction = submission["prediction"]

        df = pd.DataFrame(
            {
                "date": prediction["dates"],
                "median": prediction["preds"],
                "adm_0": submission["adm_0"],
                "adm_1": submission["adm_1"] or "BR",
            }
        )

        for lower, upper in [
            ("lower_50", "upper_50"),
            ("lower_80", "upper_80"),
            ("lower_90", "upper_90"),
            ("lower_95", "upper_95"),
        ]:
            if lower in prediction and upper in prediction:
                df[lower] = prediction[lower]
                df[upper] = prediction[upper]

        df.to_csv(filepath, index=False)

    def get_submission(self, index: int = -1) -> Dict[str, Any]:
        """Get a submission by index.

        Args:
            index: Submission index (default: last)

        Returns:
            Submission dictionary
        """
        return self.submissions[index]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all submissions to a single DataFrame.

        Returns:
            DataFrame with all forecasts
        """
        rows = []

        for submission in self.submissions:
            prediction = submission["prediction"]
            n_dates = len(prediction["dates"])

            for i in range(n_dates):
                row = {
                    "date": prediction["dates"][i],
                    "adm_0": submission["adm_0"],
                    "adm_1": submission["adm_1"] or "BR",
                    "median": prediction["preds"][i],
                }

                for lower, upper in [
                    ("lower_50", "upper_50"),
                    ("lower_80", "upper_80"),
                    ("lower_90", "upper_90"),
                    ("lower_95", "upper_95"),
                ]:
                    if lower in prediction and upper in prediction:
                        row[lower] = prediction[lower][i]
                        row[upper] = prediction[upper][i]

                rows.append(row)

        return pd.DataFrame(rows)

    def validate_submissions(self) -> List[Dict[str, Any]]:
        """Validate all submissions.

        Returns:
            List of validation issues
        """
        issues = []

        for i, submission in enumerate(self.submissions):
            prediction = submission["prediction"]

            if not prediction.get("dates"):
                issues.append({"submission": i, "issue": "No dates provided"})

            if not prediction.get("preds"):
                issues.append({"submission": i, "issue": "No predictions provided"})

            if len(prediction.get("dates", [])) != len(prediction.get("preds", [])):
                issues.append({"submission": i, "issue": "Dates and predictions length mismatch"})

            for q in ["lower_50", "lower_80", "lower_90", "lower_95"]:
                if q in prediction:
                    if any(v < 0 for v in prediction[q]):
                        issues.append({"submission": i, "issue": f"Negative values in {q}"})

            for lower, upper in [
                ("lower_50", "upper_50"),
                ("lower_80", "upper_80"),
                ("lower_90", "upper_90"),
                ("lower_95", "upper_95"),
            ]:
                if lower in prediction and upper in prediction:
                    for j, (lower_val, upper_val) in enumerate(
                        zip(prediction[lower], prediction[upper])
                    ):
                        if lower_val > upper_val:
                            issues.append(
                                {"submission": i, "issue": f"{lower} > {upper} at index {j}"}
                            )

        if issues:
            logger.warning(f"Found {len(issues)} validation issues")
        else:
            logger.info("All submissions validated successfully")

        return issues

    def clear_submissions(self) -> None:
        """Clear all stored submissions."""
        self.submissions = []


def create_forecast_dataframe(
    dates: List[str],
    median: List[float],
    lower_50: Optional[List[float]] = None,
    upper_50: Optional[List[float]] = None,
    lower_80: Optional[List[float]] = None,
    upper_80: Optional[List[float]] = None,
    lower_90: Optional[List[float]] = None,
    upper_90: Optional[List[float]] = None,
    lower_95: Optional[List[float]] = None,
    upper_95: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Create a properly formatted forecast DataFrame.

    Args:
        dates: List of dates
        median: Median predictions
        lower/upper_X: Prediction interval bounds

    Returns:
        DataFrame with forecast data
    """
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "median": median,
        }
    )

    for lower, upper in [
        ("lower_50", "upper_50"),
        ("lower_80", "upper_80"),
        ("lower_90", "upper_90"),
        ("lower_95", "upper_95"),
    ]:
        lower_vals = locals().get(lower)
        upper_vals = locals().get(upper)

        if lower_vals is not None:
            df[lower] = lower_vals
        if upper_vals is not None:
            df[upper] = upper_vals

    return df


def generate_forecast_dates(
    start_date: str,
    n_weeks: int = 52,
) -> List[str]:
    """Generate weekly forecast dates starting from a given date.

    Args:
        start_date: Start date (YYYY-MM-DD)
        n_weeks: Number of weeks to generate

    Returns:
        List of date strings
    """
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-SUN")
    return [d.strftime("%Y-%m-%d") for d in dates]
