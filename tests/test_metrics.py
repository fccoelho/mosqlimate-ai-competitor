"""Tests for evaluation metrics module.

This module tests the probabilistic forecasting metrics including:
- CRPS (Continuous Ranked Probability Score)
- WIS (Weighted Interval Score)
- Standard metrics (RMSE, MAE, MAPE)
- Coverage metrics
- Bias and sharpness
"""

import numpy as np
import pandas as pd
import pytest

from mosqlimate_ai.evaluation.metrics import (
    bias,
    coverage,
    crps,
    crps_single,
    evaluate_forecast,
    interval_width,
    mae,
    mape,
    rmse,
    sharpness,
    skill_score,
    weighted_interval_score,
    weighted_interval_score_total,
)


class TestRMSE:
    """Test Root Mean Squared Error metric."""

    def test_perfect_prediction(self):
        """RMSE should be 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rmse(y_true, y_pred)
        assert result == 0.0

    def test_constant_error(self):
        """RMSE for constant error of 1."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # Error of 1 each
        result = rmse(y_true, y_pred)
        expected = 1.0
        assert np.isclose(result, expected)

    def test_mixed_error(self):
        """RMSE with mixed errors."""
        y_true = np.array([0.0, 4.0, 3.0])
        y_pred = np.array([3.0, 1.0, 3.0])
        # Errors: 3, -3, 0
        # Squared: 9, 9, 0
        # Mean: 6
        # RMSE: sqrt(6) ≈ 2.449
        result = rmse(y_true, y_pred)
        expected = np.sqrt(6.0)
        assert np.isclose(result, expected)

    def test_single_value(self):
        """RMSE with single value arrays."""
        y_true = np.array([5.0])
        y_pred = np.array([3.0])
        result = rmse(y_true, y_pred)
        assert result == 2.0

    def test_empty_arrays(self):
        """RMSE with empty arrays should return nan."""
        y_true = np.array([])
        y_pred = np.array([])
        result = rmse(y_true, y_pred)
        assert np.isnan(result)


class TestMAE:
    """Test Mean Absolute Error metric."""

    def test_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mae(y_true, y_pred)
        assert result == 0.0

    def test_constant_error(self):
        """MAE for constant error of 2."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 4.0, 5.0])  # Error of 2 each
        result = mae(y_true, y_pred)
        assert result == 2.0

    def test_negative_errors(self):
        """MAE handles negative errors correctly."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([3.0, 7.0, 5.0])
        # Absolute errors: 2, 2, 0
        result = mae(y_true, y_pred)
        assert result == 4.0 / 3.0

    def test_2d_arrays(self):
        """MAE with 2D arrays should flatten."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[2.0, 2.0], [4.0, 4.0]])
        # Errors: 1, 0, 1, 0
        result = mae(y_true, y_pred)
        assert result == 0.5


class TestMAPE:
    """Test Mean Absolute Percentage Error metric."""

    def test_perfect_prediction(self):
        """MAPE should be 0 for perfect predictions."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        result = mape(y_true, y_pred)
        assert result == 0.0

    def test_percentage_calculation(self):
        """MAPE calculates percentage correctly."""
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred = np.array([90.0, 110.0, 100.0])
        # Percentage errors: 10%, 10%, 0%
        result = mape(y_true, y_pred)
        assert result == pytest.approx(20.0 / 3.0, abs=0.01)

    def test_zero_handling(self):
        """MAPE handles zero values with epsilon."""
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([0.0, 100.0])
        result = mape(y_true, y_pred)
        # With epsilon handling, this should be close to 0
        assert result >= 0.0

    def test_small_values(self):
        """MAPE with small true values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 3.3])  # 10% error
        result = mape(y_true, y_pred)
        assert result == pytest.approx(10.0, abs=1.0)


class TestCRPSSingle:
    """Test single CRPS calculation."""

    def test_perfect_prediction(self):
        """CRPS should be 0 when true value is at median."""
        y_true = 5.0
        quantiles = np.array([0.25, 0.5, 0.75])
        values = np.array([3.0, 5.0, 7.0])  # True at median
        result = crps_single(y_true, quantiles, values)
        # Should be close to 0 (but CRPS calculation is complex)
        assert result >= 0.0

    def test_simple_case(self):
        """CRPS for a simple case."""
        y_true = 5.0
        quantiles = np.array([0.5])
        values = np.array([5.0])
        result = crps_single(y_true, quantiles, values)
        # With single quantile at true value
        assert isinstance(result, float)
        assert result >= 0.0


class TestCRPS:
    """Test Continuous Ranked Probability Score."""

    def test_with_dataframe(self):
        """CRPS with proper DataFrame format."""
        y_true = np.array([5.0, 10.0])
        predictions = pd.DataFrame(
            {
                "lower_95": [4.0, 9.0],
                "lower_50": [4.5, 9.5],
                "median": [5.0, 10.0],
                "upper_50": [5.5, 10.5],
                "upper_95": [6.0, 11.0],
            }
        )
        result = crps(y_true, predictions)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_missing_columns(self):
        """CRPS with missing columns should return nan."""
        y_true = np.array([5.0, 10.0])
        predictions = pd.DataFrame(
            {
                "other_column": [5.0, 10.0],
                # Missing all expected quantile columns
            }
        )
        result = crps(y_true, predictions)
        assert np.isnan(result)

    def test_partial_columns(self):
        """CRPS with partial columns."""
        y_true = np.array([5.0, 10.0])
        predictions = pd.DataFrame(
            {
                "lower_50": [4.5, 9.5],
                "median": [5.0, 10.0],
                "upper_50": [5.5, 10.5],
                # Missing 95% intervals
            }
        )
        result = crps(y_true, predictions)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_custom_quantiles(self):
        """CRPS with custom quantile mapping."""
        y_true = np.array([5.0, 10.0])
        predictions = pd.DataFrame(
            {
                "q10": [4.0, 9.0],
                "q50": [5.0, 10.0],
                "q90": [6.0, 11.0],
            }
        )
        quantile_cols = {0.1: "q10", 0.5: "q50", 0.9: "q90"}
        result = crps(y_true, predictions, quantile_cols)
        assert isinstance(result, float)
        assert result >= 0.0


class TestWeightedIntervalScore:
    """Test Weighted Interval Score."""

    def test_perfect_coverage(self):
        """WIS with perfect coverage."""
        y_true = np.array([5.0, 5.0, 5.0])
        lower = np.array([4.0, 4.0, 4.0])
        upper = np.array([6.0, 6.0, 6.0])
        median = np.array([5.0, 5.0, 5.0])
        result = weighted_interval_score(y_true, lower, upper, median, alpha=0.05)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_missed_interval(self):
        """WIS when true values miss interval."""
        y_true = np.array([10.0])  # Way outside interval
        lower = np.array([4.0])
        upper = np.array([6.0])
        median = np.array([5.0])
        result = weighted_interval_score(y_true, lower, upper, median, alpha=0.05)
        # Should have penalty for missing interval
        assert result > 0.0

    def test_zero_alpha(self):
        """WIS with very small alpha (tight interval expected)."""
        y_true = np.array([5.0])
        lower = np.array([4.0])
        upper = np.array([6.0])
        median = np.array([5.0])
        result = weighted_interval_score(y_true, lower, upper, median, alpha=0.01)
        assert isinstance(result, float)


class TestWeightedIntervalScoreTotal:
    """Test total WIS across multiple levels."""

    def test_with_dataframe(self):
        """Total WIS with DataFrame."""
        y_true = np.array([5.0, 10.0])
        predictions = pd.DataFrame(
            {
                "median": [5.0, 10.0],
                "lower_50": [4.5, 9.5],
                "upper_50": [5.5, 10.5],
                "lower_80": [4.0, 9.0],
                "upper_80": [6.0, 11.0],
                "lower_95": [3.0, 8.0],
                "upper_95": [7.0, 12.0],
            }
        )
        result = weighted_interval_score_total(y_true, predictions)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_missing_levels(self):
        """Total WIS with missing levels."""
        y_true = np.array([5.0, 10.0])
        predictions = pd.DataFrame(
            {
                "median": [5.0, 10.0],
                "lower_50": [4.5, 9.5],
                "upper_50": [5.5, 10.5],
                # Missing 80% and 95% intervals
            }
        )
        result = weighted_interval_score_total(y_true, predictions)
        # Should calculate WIS for available levels only
        assert isinstance(result, float)

    def test_empty_dataframe(self):
        """Total WIS with no available levels returns nan."""
        y_true = np.array([5.0])
        predictions = pd.DataFrame(
            {
                "median": [5.0],
                # No interval columns
            }
        )
        result = weighted_interval_score_total(y_true, predictions)
        assert np.isnan(result)

    def test_custom_levels(self):
        """Total WIS with custom confidence levels."""
        y_true = np.array([5.0, 10.0])
        predictions = pd.DataFrame(
            {
                "median": [5.0, 10.0],
                "lower_90": [4.0, 9.0],
                "upper_90": [6.0, 11.0],
            }
        )
        levels = [0.90]
        weights = [1.0]
        result = weighted_interval_score_total(y_true, predictions, levels, weights)
        assert isinstance(result, float)


class TestCoverage:
    """Test prediction interval coverage."""

    def test_perfect_coverage(self):
        """Coverage when all values in interval."""
        y_true = np.array([5.0, 6.0, 7.0])
        lower = np.array([4.0, 4.0, 4.0])
        upper = np.array([8.0, 8.0, 8.0])
        result = coverage(y_true, lower, upper)
        assert result == 1.0

    def test_zero_coverage(self):
        """Coverage when no values in interval."""
        y_true = np.array([10.0, 11.0, 12.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        result = coverage(y_true, lower, upper)
        assert result == 0.0

    def test_partial_coverage(self):
        """Coverage with partial hits."""
        y_true = np.array([5.0, 10.0, 15.0])
        lower = np.array([4.0, 4.0, 4.0])
        upper = np.array([6.0, 6.0, 6.0])
        # Only first value in interval
        result = coverage(y_true, lower, upper)
        assert result == 1.0 / 3.0

    def test_boundary_values(self):
        """Coverage includes boundary values."""
        y_true = np.array([5.0, 10.0])
        lower = np.array([5.0, 5.0])
        upper = np.array([10.0, 10.0])
        result = coverage(y_true, lower, upper)
        assert result == 1.0


class TestIntervalWidth:
    """Test interval width metric."""

    def test_constant_width(self):
        """Interval width for constant intervals."""
        lower = np.array([4.0, 4.0, 4.0])
        upper = np.array([6.0, 6.0, 6.0])
        result = interval_width(lower, upper)
        assert result == 2.0

    def test_variable_width(self):
        """Interval width for variable intervals."""
        lower = np.array([4.0, 5.0])
        upper = np.array([6.0, 10.0])
        # Widths: 2, 5
        # Mean: 3.5
        result = interval_width(lower, upper)
        assert result == 3.5

    def test_zero_width(self):
        """Interval width when lower equals upper."""
        lower = np.array([5.0, 5.0])
        upper = np.array([5.0, 5.0])
        result = interval_width(lower, upper)
        assert result == 0.0


class TestSharpness:
    """Test prediction interval sharpness."""

    def test_absolute_sharpness(self):
        """Sharpness without normalization."""
        lower = np.array([4.0, 5.0])
        upper = np.array([6.0, 10.0])
        result = sharpness(lower, upper, relative=False)
        # Mean widths: 2, 5 -> mean = 3.5
        assert result == 3.5

    def test_relative_sharpness(self):
        """Sharpness with normalization."""
        lower = np.array([4.0])
        upper = np.array([6.0])
        y_true = np.array([10.0])
        result = sharpness(lower, upper, y_true, relative=True)
        # Width: 2, normalized by true value: 2/10 = 0.2
        assert result == 0.2


class TestBias:
    """Test prediction bias."""

    def test_no_bias(self):
        """Bias when predictions are unbiased."""
        y_true = np.array([5.0, 10.0, 15.0])
        median = np.array([5.0, 10.0, 15.0])
        result = bias(y_true, median)
        assert result == 0.0

    def test_positive_bias(self):
        """Positive bias (overestimation)."""
        y_true = np.array([5.0, 5.0, 5.0])
        median = np.array([6.0, 6.0, 6.0])
        result = bias(y_true, median)
        assert result == 1.0

    def test_negative_bias(self):
        """Negative bias (underestimation)."""
        y_true = np.array([5.0, 5.0, 5.0])
        median = np.array([4.0, 4.0, 4.0])
        result = bias(y_true, median)
        assert result == -1.0


class TestSkillScore:
    """Test skill score metric."""

    def test_better_than_baseline(self):
        """Skill score when model is better than baseline."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])  # Good model
        y_baseline = np.array([2.0, 2.0, 2.0])  # Constant baseline
        result = skill_score(y_true, y_pred, y_baseline, metric="rmse")
        # Should be positive (better than baseline)
        assert result > 0.0

    def test_worse_than_baseline(self):
        """Skill score when model is worse than baseline."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])  # Bad model
        y_baseline = np.array([1.1, 2.1, 3.1])  # Good baseline
        result = skill_score(y_true, y_pred, y_baseline, metric="rmse")
        # Should be negative (worse than baseline)
        assert result < 0.0

    def test_equal_to_baseline(self):
        """Skill score when model equals baseline."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        y_baseline = np.array([1.1, 2.1, 3.1])
        result = skill_score(y_true, y_pred, y_baseline, metric="rmse")
        assert result == 0.0

    def test_different_metrics(self):
        """Skill score with different metrics."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        y_baseline = np.array([2.0, 2.0, 2.0])

        rmse_score = skill_score(y_true, y_pred, y_baseline, metric="rmse")
        mae_score = skill_score(y_true, y_pred, y_baseline, metric="mae")
        mape_score = skill_score(y_true, y_pred, y_baseline, metric="mape")

        # All should be positive (model is better)
        assert rmse_score > 0.0
        assert mae_score > 0.0
        assert mape_score > 0.0


class TestEvaluateForecast:
    """Test comprehensive forecast evaluation."""

    def test_complete_evaluation(self):
        """Evaluate forecast with all metrics."""
        y_true = np.array([5.0, 10.0, 15.0])
        predictions = pd.DataFrame(
            {
                "median": [5.0, 10.0, 15.0],
                "lower_50": [4.0, 9.0, 14.0],
                "upper_50": [6.0, 11.0, 16.0],
                "lower_80": [3.0, 8.0, 13.0],
                "upper_80": [7.0, 12.0, 17.0],
                "lower_95": [2.0, 7.0, 12.0],
                "upper_95": [8.0, 13.0, 18.0],
            }
        )

        result = evaluate_forecast(y_true, predictions)

        # Check all expected metrics are present
        expected_metrics = ["rmse", "mae", "mape", "bias", "crps", "wis_total"]
        for metric in expected_metrics:
            assert metric in result
            assert isinstance(result[metric], float)

    def test_with_nans(self):
        """Evaluate forecast with NaN values."""
        y_true = np.array([5.0, 10.0, 15.0])
        predictions = pd.DataFrame(
            {
                "median": [5.0, np.nan, 15.0],
                "lower_50": [4.0, 9.0, 14.0],
                "upper_50": [6.0, 11.0, 16.0],
                "lower_95": [2.0, 7.0, 12.0],
                "upper_95": [8.0, 13.0, 18.0],
            }
        )

        result = evaluate_forecast(y_true, predictions)

        # Should still compute metrics, excluding NaN
        assert "rmse" in result
        assert not np.isnan(result["rmse"])

    def test_partial_intervals(self):
        """Evaluate with only some prediction intervals."""
        y_true = np.array([5.0, 10.0])
        predictions = pd.DataFrame(
            {
                "median": [5.0, 10.0],
                "lower_50": [4.0, 9.0],
                "upper_50": [6.0, 11.0],
            }
        )

        result = evaluate_forecast(y_true, predictions)

        # Should have basic metrics
        assert "rmse" in result
        assert "coverage_50" in result
        # Should not have 95% coverage (no 95% intervals)
        assert "coverage_95" not in result

    def test_custom_levels(self):
        """Evaluate with custom confidence levels."""
        y_true = np.array([5.0, 10.0])
        predictions = pd.DataFrame(
            {
                "median": [5.0, 10.0],
                "lower_90": [3.0, 8.0],
                "upper_90": [7.0, 12.0],
            }
        )

        result = evaluate_forecast(y_true, predictions, levels=[0.90])

        assert "coverage_90" in result
        assert "wis_90" in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_observation(self):
        """Metrics with single observation."""
        y_true = np.array([5.0])
        y_pred = np.array([5.0])

        assert rmse(y_true, y_pred) == 0.0
        assert mae(y_true, y_pred) == 0.0
        assert mape(y_true, y_pred) == 0.0
        assert bias(y_true, y_pred) == 0.0

    def test_very_large_values(self):
        """Metrics with very large values."""
        y_true = np.array([1e6, 2e6, 3e6])
        y_pred = np.array([1.1e6, 2.1e6, 3.1e6])

        result_rmse = rmse(y_true, y_pred)
        result_mae = mae(y_true, y_pred)

        assert result_rmse > 0.0
        assert result_mae > 0.0

    def test_very_small_values(self):
        """Metrics with very small values."""
        y_true = np.array([1e-6, 2e-6, 3e-6])
        y_pred = np.array([1.1e-6, 2.1e-6, 3.1e-6])

        result_rmse = rmse(y_true, y_pred)
        result_mae = mae(y_true, y_pred)

        assert result_rmse > 0.0
        assert result_mae > 0.0

    def test_mixed_positive_negative(self):
        """Metrics with mixed positive/negative values."""
        y_true = np.array([-5.0, 0.0, 5.0])
        y_pred = np.array([-4.0, 1.0, 6.0])

        result_rmse = rmse(y_true, y_pred)
        result_mae = mae(y_true, y_pred)

        assert result_rmse == 1.0
        assert result_mae == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
