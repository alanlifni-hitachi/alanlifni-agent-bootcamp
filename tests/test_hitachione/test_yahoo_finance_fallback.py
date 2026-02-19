"""Tests for the Yahoo Finance fallback module."""

from unittest.mock import MagicMock, patch

import pytest


class TestYFinanceFallback:
    """Tests for yahoo_finance_fallback helpers."""

    @patch("src.hitachione.tools.yahoo_finance_fallback._safe_import_yfinance")
    def test_get_yf_performance_data_returns_correct_shape(self, mock_import):
        """Should return dict with price_data, earnings, news keys."""
        import pandas as pd

        # Mock yfinance Ticker
        mock_yf = MagicMock()
        mock_import.return_value = mock_yf

        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        # Mock history (DataFrame)
        mock_ticker.history.return_value = pd.DataFrame({
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 100.0],
            "Close": [104.0, 105.0],
            "Volume": [1000000, 1100000],
        }, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))

        mock_ticker.info = {
            "shortName": "ExxonMobil",
            "sector": "Energy",
            "industry": "Oil & Gas Integrated",
            "marketCap": 400_000_000_000,
            "trailingPE": 12.5,
        }
        mock_ticker.news = []

        from src.hitachione.tools.yahoo_finance_fallback import get_yf_performance_data

        result = get_yf_performance_data("XOM")

        assert "price_data" in result
        assert "earnings" in result
        assert "news" in result
        assert len(result["price_data"]) == 2
        assert result["price_data"][0]["close"] == 104.0
        assert "text" in result["price_data"][0]
        assert len(result["earnings"]) == 1
        assert "ExxonMobil" in result["earnings"][0]["text"]

    @patch("src.hitachione.tools.yahoo_finance_fallback._safe_import_yfinance")
    def test_get_yf_sentiment_records_returns_correct_shape(self, mock_import):
        """Should return a list of dicts with text, title, date, etc."""
        import pandas as pd

        mock_yf = MagicMock()
        mock_import.return_value = mock_yf

        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        mock_ticker.info = {
            "shortName": "Chevron",
            "sector": "Energy",
            "marketCap": 300_000_000_000,
        }
        mock_ticker.news = [
            {
                "content": {
                    "title": "Chevron Reports Strong Q4",
                    "summary": "Chevron beat estimates with strong drilling output.",
                    "pubDate": "2024-02-01T12:00:00Z",
                }
            }
        ]
        mock_ticker.history.return_value = pd.DataFrame({
            "Open": [150.0],
            "High": [155.0],
            "Low": [149.0],
            "Close": [153.0],
            "Volume": [2000000],
        }, index=pd.to_datetime(["2024-02-01"]))

        from src.hitachione.tools.yahoo_finance_fallback import get_yf_sentiment_records

        records = get_yf_sentiment_records("CVX")

        assert len(records) >= 2  # company info + news + price action
        assert any("Chevron" in r.get("text", "") for r in records)
        assert any(r.get("dataset_source") == "yahoo_finance_live" for r in records)

    @patch("src.hitachione.tools.yahoo_finance_fallback._safe_import_yfinance")
    def test_fallback_handles_exception_gracefully(self, mock_import):
        """If yfinance raises, return empty data instead of crashing."""
        mock_yf = MagicMock()
        mock_import.return_value = mock_yf
        mock_yf.Ticker.side_effect = Exception("API rate limit exceeded")

        from src.hitachione.tools.yahoo_finance_fallback import (
            get_yf_performance_data,
            get_yf_sentiment_records,
        )

        perf = get_yf_performance_data("FAIL")
        assert perf == {"price_data": [], "earnings": [], "news": []}

        sent = get_yf_sentiment_records("FAIL")
        assert sent == []

    def test_ticker_info_summary_format(self):
        """_ticker_info_summary should produce a readable string."""
        from src.hitachione.tools.yahoo_finance_fallback import _ticker_info_summary

        info = {
            "shortName": "ExxonMobil",
            "sector": "Energy",
            "industry": "Oil & Gas Integrated",
            "marketCap": 400_000_000_000,
            "trailingPE": 12.5,
            "dividendYield": 0.035,
        }
        summary = _ticker_info_summary(info, "XOM")
        assert "ExxonMobil" in summary
        assert "Energy" in summary
        assert "$400.00B" in summary
        assert "3.50%" in summary


class TestParseTimeframe:
    """Tests for _parse_timeframe helper."""

    def test_pure_year(self):
        from src.hitachione.tools.yahoo_finance_fallback import _parse_timeframe
        start, end = _parse_timeframe("2012")
        assert start == "2012-01-01"
        assert end == "2012-12-31"

    def test_quarter(self):
        from src.hitachione.tools.yahoo_finance_fallback import _parse_timeframe
        start, end = _parse_timeframe("2024 Q3")
        assert start == "2024-07-01"
        assert end == "2024-09-30"

    def test_quarter_reversed(self):
        from src.hitachione.tools.yahoo_finance_fallback import _parse_timeframe
        start, end = _parse_timeframe("Q1 2023")
        assert start == "2023-01-01"
        assert end == "2023-03-31"

    def test_empty_returns_none(self):
        from src.hitachione.tools.yahoo_finance_fallback import _parse_timeframe
        assert _parse_timeframe("") == (None, None)
        assert _parse_timeframe(None) == (None, None)

    def test_unrecognised_returns_none(self):
        from src.hitachione.tools.yahoo_finance_fallback import _parse_timeframe
        assert _parse_timeframe("sometime recently") == (None, None)
