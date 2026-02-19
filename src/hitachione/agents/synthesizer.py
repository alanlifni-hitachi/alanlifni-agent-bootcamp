"""Synthesizer Agent – composes a ranked / comparative answer from research.

Takes a list of ``CompanyResearch`` objects and produces a user-facing
``SynthesizedAnswer`` with:
  • Markdown answer (ranked list or comparison table)
  • Rationale explaining the scoring / ranking
  • Caveats for partial data
  • Citations from news snippets
  • Confidence estimate
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from ..config.settings import OPENAI_BASE_URL, OPENAI_API_KEY, WORKER_MODEL
from ..models.schemas import (
    CompanyResearch, Intent, SynthesizedAnswer, TaskContext,
)

logger = logging.getLogger(__name__)

# ── Prompt templates ─────────────────────────────────────────────────────

_SYSTEM = (
    "You are a financial intelligence synthesizer. "
    "Given structured research data for companies, compose a clear, "
    "well-structured Markdown answer for the user. "
    "Include a rationale section explaining your reasoning. "
    "If data is partial, add explicit caveats. "
    "If news references are available, include them as citations. "
    "If period stats are provided for a requested timeframe, treat those stats "
    "as authoritative and do not claim missing data for that ticker/timeframe. "
    "Do NOT give investment advice – present analytics and insights only. "
    "Return raw Markdown text (not wrapped in JSON)."
)


def _build_research_block(research: list[CompanyResearch]) -> str:
    """Serialize research data into a text block for the LLM."""
    parts: list[str] = []
    for cr in research:
        lines = [f"## {cr.ticker}"]
        if cr.sentiment:
            lines.append(
                f"- Sentiment: rating={cr.sentiment.get('rating')}, "
                f"label={cr.sentiment.get('label')}, "
                f"rationale={cr.sentiment.get('rationale','')[:200]}"
            )
        if cr.performance:
            lines.append(
                f"- Performance: score={cr.performance.get('performance_score')}, "
                f"outlook={cr.performance.get('outlook')}, "
                f"justification={cr.performance.get('justification','')[:200]}"
            )
            perf_summary = cr.performance.get("data_summary", {})
            period_stats = perf_summary.get("period_stats", {}) if isinstance(perf_summary, dict) else {}
            if isinstance(period_stats, dict) and period_stats:
                lines.append(
                    "- Period stats: "
                    f"start={period_stats.get('start_date')}, "
                    f"end={period_stats.get('end_date')}, "
                    f"first_close={period_stats.get('first_close')}, "
                    f"last_close={period_stats.get('last_close')}, "
                    f"percent_change={period_stats.get('percent_change')}, "
                    f"high={period_stats.get('period_high')}, "
                    f"low={period_stats.get('period_low')}, "
                    f"trading_days={period_stats.get('trading_days')}"
                )
        if cr.news_snippets:
            lines.append("- News references: " + "; ".join(cr.news_snippets[:3]))
        if cr.errors:
            lines.append(
                "- ⚠ Data gaps: "
                + ", ".join(f"{e.tool} ({e.error[:60]})" for e in cr.errors)
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _estimate_confidence(research: list[CompanyResearch]) -> float:
    """Heuristic confidence score 0-1 based on data completeness."""
    if not research:
        return 0.0
    scores: list[float] = []
    for cr in research:
        s = 0.0
        if cr.sentiment and cr.sentiment.get("rating") is not None:
            s += 0.5
        if cr.performance and cr.performance.get("performance_score") is not None:
            s += 0.5
        scores.append(s)
    return sum(scores) / len(scores)


def _extract_company_provenance(cr: CompanyResearch) -> tuple[list[str], bool, bool, str]:
    """Extract dataset sources and fallback usage from one company bundle."""
    sources: set[str] = set()
    yahoo_fallback = False
    yahoo_fallback_attempted = False
    fallback_errors: set[str] = set()

    for payload in (cr.sentiment, cr.performance):
        if not payload:
            continue
        summary = payload.get("data_summary", {})
        if isinstance(summary, dict):
            dataset_sources = summary.get("dataset_sources", []) or []
            for ds in dataset_sources:
                if ds:
                    sources.add(str(ds))

            source_name = str(summary.get("source", ""))
            if source_name and not dataset_sources:
                sources.add(source_name)

            fallback_used = summary.get("yahoo_finance_fallback_used")
            if isinstance(fallback_used, bool):
                yahoo_fallback = yahoo_fallback or fallback_used
            elif "yahoo" in source_name.lower():
                yahoo_fallback = True

            fallback_attempted = summary.get("yahoo_finance_fallback_attempted")
            if isinstance(fallback_attempted, bool):
                yahoo_fallback_attempted = yahoo_fallback_attempted or fallback_attempted

            fallback_error = str(summary.get("yahoo_finance_fallback_error", "")).strip()
            if fallback_error:
                fallback_errors.add(fallback_error)

    if yahoo_fallback:
        yahoo_fallback_attempted = True

    error_text = "; ".join(sorted(fallback_errors))
    return sorted(sources), yahoo_fallback, yahoo_fallback_attempted, error_text


def _format_data_used_section(answer: SynthesizedAnswer) -> str:
    """Render a markdown section listing data sources and fallback usage."""
    lines = ["---", "**Data used for inference**"]

    if not answer.per_ticker_data_sources:
        lines.append("- No source metadata available.")
        return "\n".join(lines)

    for ticker, sources in answer.per_ticker_data_sources.items():
        source_text = ", ".join(sources) if sources else "unknown"
        fallback_used = answer.per_ticker_yahoo_fallback.get(ticker, False)
        fallback_attempted = answer.per_ticker_yahoo_fallback_attempted.get(ticker, False)
        fallback_error = answer.per_ticker_yahoo_fallback_error.get(ticker, "")
        if fallback_used:
            fallback_status = "used"
        elif fallback_attempted and fallback_error:
            fallback_status = f"failed ({fallback_error[:120]})"
        elif fallback_attempted:
            fallback_status = "attempted_not_used"
        else:
            fallback_status = "not_attempted"
        lines.append(
            f"- {ticker}: datasets={source_text}; "
            f"yahoo_finance_fallback={fallback_status}"
        )

    lines.append(
        "- Any Yahoo Finance fallback used: "
        f"{'yes' if answer.yahoo_finance_fallback_used else 'no'}"
    )
    return "\n".join(lines)


def _format_timeframe_summary(
    ctx: TaskContext,
    research: list[CompanyResearch],
) -> str:
    """Render deterministic historical-period metrics when timeframe is provided."""
    if not (ctx.timeframe or "").strip():
        return ""

    lines = ["---", f"**Historical period summary ({ctx.timeframe})**"]
    added = 0

    for cr in research:
        summary = cr.performance.get("data_summary", {}) if cr.performance else {}
        stats = summary.get("period_stats", {}) if isinstance(summary, dict) else {}
        if not isinstance(stats, dict) or not stats:
            continue

        start = stats.get("start_date") or "unknown"
        end = stats.get("end_date") or "unknown"
        first_close = stats.get("first_close")
        last_close = stats.get("last_close")
        pct = stats.get("percent_change")
        period_high = stats.get("period_high")
        period_low = stats.get("period_low")
        trading_days = stats.get("trading_days")
        pct_text = f"{pct:+.2f}%" if isinstance(pct, (int, float)) else "n/a"
        first_close_text = (
            f"${first_close:.2f}" if isinstance(first_close, (int, float)) else "n/a"
        )
        last_close_text = (
            f"${last_close:.2f}" if isinstance(last_close, (int, float)) else "n/a"
        )
        high_text = f"${period_high:.2f}" if isinstance(period_high, (int, float)) else "n/a"
        low_text = f"${period_low:.2f}" if isinstance(period_low, (int, float)) else "n/a"
        days_text = str(trading_days) if trading_days is not None else "n/a"

        lines.append(
            f"- {cr.ticker}: {start} to {end}; "
            f"close {first_close_text} -> {last_close_text} "
            f"({pct_text})"
        )
        lines.append(
            f"- {cr.ticker}: high={high_text}, low={low_text}, "
            f"trading_days={days_text}"
        )
        added += 1

    if added == 0:
        lines.append("- No historical price-series metrics available for this timeframe.")
    return "\n".join(lines)


class SynthesizerAgent:
    """Compose a user-facing answer from per-entity research."""

    def __init__(self):
        self._llm = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    def run(
        self,
        ctx: TaskContext,
        research: list[CompanyResearch],
    ) -> SynthesizedAnswer:
        answer = SynthesizedAnswer(raw_research=research)
        answer.confidence = _estimate_confidence(research)

        # Gather provenance metadata
        all_sources: set[str] = set()
        for cr in research:
            (
                sources,
                yahoo_fallback,
                yahoo_fallback_attempted,
                fallback_error,
            ) = _extract_company_provenance(cr)
            answer.per_ticker_data_sources[cr.ticker] = sources
            answer.per_ticker_yahoo_fallback[cr.ticker] = yahoo_fallback
            answer.per_ticker_yahoo_fallback_attempted[cr.ticker] = (
                yahoo_fallback_attempted
            )
            answer.per_ticker_yahoo_fallback_error[cr.ticker] = fallback_error
            all_sources.update(sources)
            answer.yahoo_finance_fallback_used = (
                answer.yahoo_finance_fallback_used or yahoo_fallback
            )
        answer.data_sources = sorted(all_sources)

        # Gather caveats
        for cr in research:
            for e in cr.errors:
                answer.caveats.append(f"{e.entity}: {e.tool} unavailable ({e.error[:80]})")
        if answer.confidence < 0.5:
            answer.caveats.append("Low overall data coverage – results are best-effort.")

        # Gather citations
        for cr in research:
            answer.citations.extend(cr.news_snippets[:3])

        # Build the research context for the LLM
        data_block = _build_research_block(research)
        user_msg = (
            f"User query: {ctx.user_query}\n"
            f"Intent: {ctx.intent.value}\n"
            f"Timeframe: {ctx.timeframe or 'not specified'}\n\n"
            f"Research data:\n{data_block}"
        )

        try:
            resp = self._llm.chat.completions.create(
                model=WORKER_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )
            md = (resp.choices[0].message.content or "").strip()
            answer.markdown = md
            # Extract first paragraph as rationale summary
            answer.rationale = md.split("\n\n")[0] if md else ""
        except Exception as exc:
            logger.error("Synthesizer LLM error: %s", exc)
            # Fallback: build a simple text answer from raw data
            answer.markdown = self._fallback_markdown(ctx, research)
            answer.rationale = "Generated from raw data (LLM unavailable)."

        answer.markdown = "\n\n".join([
            answer.markdown.rstrip(),
            _format_timeframe_summary(ctx, research),
            _format_data_used_section(answer),
        ]).strip()

        return answer

    @staticmethod
    def _fallback_markdown(
        ctx: TaskContext, research: list[CompanyResearch]
    ) -> str:
        lines = [f"## Results for: {ctx.user_query}\n"]
        for cr in research:
            sent = cr.sentiment.get("rating", "?") if cr.sentiment else "?"
            perf = cr.performance.get("performance_score", "?") if cr.performance else "?"
            lines.append(f"- **{cr.ticker}**: sentiment={sent}/10, performance={perf}/10")
        return "\n".join(lines)
