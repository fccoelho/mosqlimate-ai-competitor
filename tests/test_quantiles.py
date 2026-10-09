"""Tests for widen_quantiles (post-hoc interval recalibration)."""

import numpy as np
import pandas as pd
import pytest

from mosqlimate_ai.evaluation.quantiles import QUANTILE_COLS, widen_quantiles


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-10-04", periods=3, freq="7D"),
            "q025": [0.0, 1.0, 2.0],
            "q050": [1.0, 2.0, 4.0],
            "q100": [2.0, 3.0, 6.0],
            "q250": [3.0, 4.0, 8.0],
            "q500": [5.0, 6.0, 10.0],
            "q750": [7.0, 8.0, 12.0],
            "q900": [8.0, 9.0, 14.0],
            "q950": [9.0, 10.0, 16.0],
            "q975": [10.0, 11.0, 18.0],
        }
    )


class TestWidenQuantiles:
    def test_identity_at_k1(self):
        df = make_frame()
        pd.testing.assert_frame_equal(widen_quantiles(df, 1.0), df)

    def test_spread_scales_around_median(self):
        out = widen_quantiles(make_frame(), 2.0)
        # q025 row0: 5 + 2*(0-5) = -5 -> clipped to 0
        assert out.loc[0, "q025"] == 0.0
        # q975 row0: 5 + 2*(10-5) = 15
        assert out.loc[0, "q975"] == pytest.approx(15.0)
        # median never moves
        assert (out["q500"] == [5.0, 6.0, 10.0]).all()

    def test_monotonicity_preserved(self):
        out = widen_quantiles(make_frame(), 3.0)
        cols = [QUANTILE_COLS[t] for t in sorted(QUANTILE_COLS)]
        assert (np.diff(out[cols].values, axis=1) >= -1e-12).all()

    def test_non_quantile_columns_pass_through(self):
        df = make_frame()
        out = widen_quantiles(df, 1.5)
        assert (out["date"] == df["date"]).all()

    def test_tightening(self):
        out = widen_quantiles(make_frame(), 0.5)
        # q975 row0: 5 + 0.5*(10-5) = 7.5
        assert out.loc[0, "q975"] == pytest.approx(7.5)
        assert out.loc[0, "q025"] == pytest.approx(2.5)
