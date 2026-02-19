"""Yahoo Finance fallback – live data for tickers absent from the Weaviate KB.

When the Weaviate knowledge base returns *no records* for a ticker (e.g. oil
stocks like XOM, CVX), this module fetches real-time / recent data directly
from Yahoo Finance via the ``yfinance`` library and formats it so the existing
LLM analysis prompts can consume it unchanged.

Public helpers
--------------
``get_yf_performance_data(ticker)``
    Returns a ``dict[str, list[dict]]`` with keys ``price_data``, ``earnings``,
    ``news`` – the same shape as ``performance_analysis_tool.get_ticker_data``.

``get_yf_sentiment_records(ticker)``
    Returns a ``list[dict]`` of records – the same shape as
    ``sentiment_analysis_tool.query_weaviate_by_ticker``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_import_yfinance():
    """Import yfinance lazily so the module doesn't crash at import time."""
    try:
        import yfinance as yf
        return yf
    except ImportError:
        logger.error(
            "yfinance is not installed. Run `pip install yfinance` to enable "
            "the Yahoo Finance fallback."
        )
        raise


def _parse_timeframe(timeframe: str) -> tuple[str | None, str | None]:
    """Convert a human timeframe string into (start_date, end_date) for yfinance.

    Supported patterns:
      - "2012"  → ("2012-01-01", "2012-12-31")
      - "2024 Q3" / "Q3 2024" → ("2024-07-01", "2024-09-30")
      - "last 3 years" → (3 years ago, today)
      - "" / None → (None, None)  → caller uses default period
    """
    if not timeframe or not timeframe.strip():
        return None, None

    tf = timeframe.strip()

    # Pure year: "2012", "2023"
    m = re.fullmatch(r"(\d{4})", tf)
    if m:
        year = m.group(1)
        return f"{year}-01-01", f"{year}-12-31"

    # Quarter: "2024 Q3", "Q3 2024", "2024Q3"
    m = re.search(r"(\d{4})\s*Q([1-4])", tf, re.IGNORECASE)
    if not m:
        m = re.search(r"Q([1-4])\s*(\d{4})", tf, re.IGNORECASE)
        if m:
            quarter, year = int(m.group(1)), m.group(2)
        else:
            quarter, year = None, None
    else:
        year, quarter = m.group(1), int(m.group(2))

    if year and quarter:
        q_starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
        q_ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
        return f"{year}-{q_starts[quarter]}", f"{year}-{q_ends[quarter]}"

    # "last N years/months"
    m = re.search(r"last\s+(\d+)\s+(year|month)", tf, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        now = datetime.now(timezone.utc)
        if unit.startswith("year"):
            start = now.replace(year=now.year - n)
        else:
            month = now.month - n
            year_offset = 0
            while month < 1:
                month += 12
                year_offset -= 1
            start = now.replace(year=now.year + year_offset, month=month)
        return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")

    # Date range: "2012-01-01 to 2012-12-31"
    m = re.search(r"(\d{4}-\d{2}-\d{2}).*?(\d{4}-\d{2}-\d{2})", tf)
    if m:
        return m.group(1), m.group(2)

    # Unrecognised – return None so caller uses defaults
    logger.debug("Could not parse timeframe '%s' – using default period", tf)
    return None, None


def _ticker_info_summary(info: dict[str, Any], ticker: str) -> str:
    """Build a concise text summary from yfinance Ticker.info."""
    parts: list[str] = []
    name = info.get("shortName") or info.get("longName") or ticker
    parts.append(f"Company: {name} ({ticker})")
    if info.get("sector"):
        parts.append(f"Sector: {info['sector']}")
    if info.get("industry"):
        parts.append(f"Industry: {info['industry']}")
    if info.get("marketCap"):
        mc = info["marketCap"]
        if mc >= 1e12:
            parts.append(f"Market Cap: ${mc/1e12:.2f}T")
        elif mc >= 1e9:
            parts.append(f"Market Cap: ${mc/1e9:.2f}B")
        else:
            parts.append(f"Market Cap: ${mc/1e6:.0f}M")
    for key, label in [
        ("trailingPE", "Trailing P/E"),
        ("forwardPE", "Forward P/E"),
        ("dividendYield", "Dividend Yield"),
        ("fiftyTwoWeekHigh", "52-Week High"),
        ("fiftyTwoWeekLow", "52-Week Low"),
        ("recommendationKey", "Analyst Recommendation"),
    ]:
        val = info.get(key)
        if val is not None:
            if key == "dividendYield":
                parts.append(f"{label}: {val:.2%}")
            else:
                parts.append(f"{label}: {val}")
    return ". ".join(parts) + "."


def _format_price_rows(hist_df) -> list[dict[str, Any]]:
    """Convert a yfinance history DataFrame to the Weaviate-like dict list."""
    rows: list[dict[str, Any]] = []
    for ts, row in hist_df.iterrows():
        date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
        entry: dict[str, Any] = {
            "date": date_str,
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        }
        entry["text"] = (
            f"On {date_str}, the stock opened at ${entry['open']:.2f}, "
            f"traded between ${entry['low']:.2f} and ${entry['high']:.2f}, "
            f"and closed at ${entry['close']:.2f} on volume of "
            f"{entry['volume']:,}."
        )
        rows.append(entry)
    return rows


def _format_news_items(news_list: list[dict], ticker: str) -> list[dict[str, Any]]:
    """Convert yfinance news dicts to Weaviate-like record dicts."""
    items: list[dict[str, Any]] = []
    for raw in news_list[:10]:  # cap at 10 items
        content = raw.get("content", {})
        title = content.get("title", "")
        summary = content.get("summary", "")
        pub = content.get("pubDate", "")
        date_str = pub[:10] if pub else ""
        text = summary or title
        if not text:
            continue
        items.append({
            "date": date_str,
            "title": title,
            "text": text,
            "category": "Yahoo Finance News",
            "dataset_source": "yahoo_finance_live",
            "ticker": ticker.upper(),
        })
    return items


# ---------------------------------------------------------------------------
# Public API – Performance fallback
# ---------------------------------------------------------------------------

def get_yf_performance_data(
    ticker: str,
    timeframe: str = "",
) -> dict[str, list[dict]]:
    """Fetch price history + news from Yahoo Finance for *ticker*.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    timeframe : str
        Optional human-readable timeframe (e.g. ``"2012"``, ``"2024 Q3"``,
        ``"last 3 years"``).  When provided, historical data for that period
        is fetched instead of the default last-1-year window.

    Returns the same structure as
    ``performance_analysis_tool.get_ticker_data``:
    ``{"price_data": [...], "earnings": [...], "news": [...]}``.
    """
    yf = _safe_import_yfinance()

    try:
        t = yf.Ticker(ticker.upper())

        # Determine date range
        start, end = _parse_timeframe(timeframe)
        if start and end:
            hist = t.history(start=start, end=end)
        else:
            hist = t.history(period="1y")
        price_data = _format_price_rows(hist)

        # Company summary as pseudo-earnings record
        info = t.info or {}
        earnings: list[dict[str, Any]] = []
        if info:
            earnings.append({
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "text": _ticker_info_summary(info, ticker),
                "title": f"{info.get('shortName', ticker)} – Company Overview",
                "dataset_source": "yahoo_finance_live",
            })

        # Recent news
        raw_news = t.news or []
        news = _format_news_items(raw_news, ticker)

        logger.info(
            "YF fallback for %s: %d price rows, %d earnings, %d news",
            ticker, len(price_data), len(earnings), len(news),
        )

        return {
            "price_data": price_data,
            "earnings": earnings,
            "news": news,
        }

    except Exception as exc:
        logger.warning("Yahoo Finance fallback failed for %s: %s", ticker, exc)
        return {"price_data": [], "earnings": [], "news": []}


# ---------------------------------------------------------------------------
# Public API – Sentiment fallback
# ---------------------------------------------------------------------------

def get_yf_sentiment_records(
    ticker: str,
    timeframe: str = "",
) -> list[dict[str, Any]]:
    """Fetch Yahoo Finance data for *ticker* as sentiment-tool-shaped records.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    timeframe : str
        Optional human-readable timeframe.  When historical, price data is
        fetched for the specified period.

    Returns the same shape as ``sentiment_analysis_tool.query_weaviate_by_ticker``.
    """
    yf = _safe_import_yfinance()

    try:
        t = yf.Ticker(ticker.upper())
        records: list[dict[str, Any]] = []

        # Company info as a single record
        info = t.info or {}
        if info:
            records.append({
                "text": _ticker_info_summary(info, ticker),
                "title": f"{info.get('shortName', ticker)} – Company Overview",
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "category": "Company Info",
                "dataset_source": "yahoo_finance_live",
                "ticker": ticker.upper(),
            })

        # Recent news (always current – yfinance doesn't have historical news)
        raw_news = t.news or []
        records.extend(_format_news_items(raw_news, ticker))

        # Price action – historical or recent depending on timeframe
        start, end = _parse_timeframe(timeframe)
        if start and end:
            hist = t.history(start=start, end=end)
            period_label = f"{start} to {end}"
        else:
            hist = t.history(period="1mo")
            period_label = "1-month"

        if not hist.empty:
            recent = hist.tail(5)
            first_close = float(hist.iloc[0]["Close"])
            last_close = float(hist.iloc[-1]["Close"])
            pct = ((last_close - first_close) / first_close) * 100
            records.append({
                "text": (
                    f"{ticker.upper()} {period_label} price change: "
                    f"${first_close:.2f} → ${last_close:.2f} ({pct:+.1f}%). "
                    f"Last 5 trading days in range: "
                    + ", ".join(
                        f"${float(row['Close']):.2f}"
                        for _, row in recent.iterrows()
                    )
                ),
                "title": f"{ticker.upper()} – Price Action ({period_label})",
                "date": hist.index[-1].strftime("%Y-%m-%d") if hasattr(hist.index[-1], 'strftime') else str(hist.index[-1])[:10],
                "category": "Price Data",
                "dataset_source": "yahoo_finance_live",
                "ticker": ticker.upper(),
            })

            # For historical periods, add monthly summary records for richer context
            if start and end and len(hist) > 20:
                for month_name, month_df in hist.groupby(hist.index.to_period('M')):
                    if len(month_df) < 2:
                        continue
                    m_open = float(month_df.iloc[0]["Open"])
                    m_close = float(month_df.iloc[-1]["Close"])
                    m_high = float(month_df["High"].max())
                    m_low = float(month_df["Low"].min())
                    m_vol = int(month_df["Volume"].sum())
                    m_pct = ((m_close - m_open) / m_open) * 100
                    records.append({
                        "text": (
                            f"{ticker.upper()} {month_name}: "
                            f"Open ${m_open:.2f}, Close ${m_close:.2f} ({m_pct:+.1f}%), "
                            f"High ${m_high:.2f}, Low ${m_low:.2f}, "
                            f"Volume {m_vol:,}"
                        ),
                        "title": f"{ticker.upper()} – Monthly Summary ({month_name})",
                        "date": month_df.index[-1].strftime("%Y-%m-%d"),
                        "category": "Price Data",
                        "dataset_source": "yahoo_finance_live",
                        "ticker": ticker.upper(),
                    })

        logger.info(
            "YF sentiment fallback for %s: %d records", ticker, len(records),
        )
        return records

    except Exception as exc:
        logger.warning(
            "Yahoo Finance sentiment fallback failed for %s: %s", ticker, exc,
        )
        return []
