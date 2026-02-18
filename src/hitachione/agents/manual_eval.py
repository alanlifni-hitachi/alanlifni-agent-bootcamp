"""Manual multi-perspective evaluator for HitachiOne orchestration.

Run from repo root:

python -m src.hitachione.agents.manual_eval
python -m src.hitachione.agents.manual_eval --quick
python -m src.hitachione.agents.manual_eval --query "Compare AAPL vs MSFT in 2024"
python -m src.hitachione.agents.manual_eval --save reports/hitachione-manual-eval.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config.settings import WEAVIATE_API_KEY, WEAVIATE_COLLECTION, WEAVIATE_HTTP_HOST
from .orchestrator import Orchestrator


DEFAULT_SCENARIOS: list[dict[str, str]] = [
    {
        "name": "rank_sector",
        "query": "Rank top 3 semiconductor stocks for 2024 and explain why.",
    },
    {
        "name": "compare_two",
        "query": "Compare AAPL and MSFT on sentiment and performance in 2024.",
    },
    {
        "name": "snapshot_single",
        "query": "Give me a snapshot of TSLA right now from available data.",
    },
    {
        "name": "event_reaction",
        "query": "How did NVDA react to recent earnings news?",
    },
    {
        "name": "ambiguous",
        "query": "What is the best stock?",
    },
    {
        "name": "invalid_entity",
        "query": "Analyze ZZZZZ and rank it against AAPL.",
    },
]

QUICK_SCENARIOS = DEFAULT_SCENARIOS[:3]


_INFRA_ERROR_PATTERNS = (
    "insufficient permissions",
    "forbidden action",
    "permission denied",
    "rbac",
)


def _weaviate_preflight() -> tuple[bool, str]:
    """Check if Weaviate collection is readable for current credentials."""
    try:
        import weaviate
        from weaviate.auth import AuthApiKey

        if WEAVIATE_HTTP_HOST.endswith(".weaviate.cloud"):
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=f"https://{WEAVIATE_HTTP_HOST}",
                auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
            )
        else:
            client = weaviate.connect_to_custom(
                http_host=WEAVIATE_HTTP_HOST,
                http_port=443,
                http_secure=True,
                grpc_host=WEAVIATE_HTTP_HOST,
                grpc_port=443,
                grpc_secure=True,
                auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
            )

        try:
            col = client.collections.get(WEAVIATE_COLLECTION)
            col.query.fetch_objects(limit=1, return_properties=["title"])
        finally:
            client.close()

        return True, ""
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if any(pattern in lowered for pattern in _INFRA_ERROR_PATTERNS):
            return False, message
        # Non-RBAC failure is still a blocker for this eval context
        return False, message


def _issue(condition: bool, message: str, issues: list[str]) -> None:
    if condition:
        issues.append(message)


def _detect_issues(result: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    confidence = float(result.get("confidence", 0.0) or 0.0)
    markdown = str(result.get("markdown", "") or "")
    caveats = result.get("caveats", []) or []
    raw_research = result.get("raw_research", []) or []

    _issue(len(markdown.strip()) < 80, "Answer markdown is unexpectedly short", issues)
    _issue(not raw_research, "No company research objects returned", issues)
    _issue(confidence > 0.75 and len(caveats) > 2, "High confidence with many caveats", issues)
    _issue(confidence < 0.25 and len(caveats) == 0, "Low confidence without caveats", issues)

    # Per-company checks
    for item in raw_research:
        ticker = item.get("ticker", "UNKNOWN")
        sentiment = item.get("sentiment", {}) or {}
        performance = item.get("performance", {}) or {}
        errors = item.get("errors", []) or []

        _issue(
            not sentiment and not performance and not errors,
            f"{ticker}: empty research payload",
            issues,
        )

        sent_rating = sentiment.get("rating")
        perf_score = performance.get("performance_score")

        _issue(
            sent_rating is not None and not (1 <= int(sent_rating) <= 10),
            f"{ticker}: sentiment rating outside 1-10",
            issues,
        )
        _issue(
            perf_score is not None and not (1 <= int(perf_score) <= 10),
            f"{ticker}: performance score outside 1-10",
            issues,
        )

    return issues


def _detect_infra_blockers(result: dict[str, Any]) -> list[str]:
    """Detect infra-level blockers (e.g. RBAC) from answer payloads."""
    findings: list[str] = []

    caveats = result.get("caveats", []) or []
    markdown = str(result.get("markdown", "") or "")
    raw_research = result.get("raw_research", []) or []

    haystacks: list[str] = [markdown, *[str(c) for c in caveats]]

    for item in raw_research:
        for err in item.get("errors", []) or []:
            haystacks.append(str(err.get("error", "")))

    for text in haystacks:
        lowered = text.lower()
        for pattern in _INFRA_ERROR_PATTERNS:
            if pattern in lowered:
                findings.append(f"Infrastructure blocker detected: {pattern}")
                break

    # De-duplicate while preserving order
    return list(dict.fromkeys(findings))


def _run_one(
    orch: Orchestrator,
    scenario: dict[str, str],
    preflight_ok: bool,
    preflight_error: str,
) -> dict[str, Any]:
    name = scenario["name"]
    query = scenario["query"]

    print(f"\n=== Scenario: {name} ===")
    print(f"Query: {query}")

    try:
        answer = orch.run(query, metadata={"eval_scenario": name})
        result = asdict(answer)
        issues = _detect_issues(result)
        blockers = _detect_infra_blockers(result)

        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"Caveats: {len(result.get('caveats', []))}")
        print(f"Companies researched: {len(result.get('raw_research', []))}")
        if blockers:
            print("Infrastructure blockers:")
            for blocker in blockers:
                print(f"  - {blocker}")

        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Issues: none")

        # If we are infra-blocked, mark scenario as blocked rather than failed
        is_blocked = len(blockers) > 0
        if (
            not preflight_ok
            and "No company research objects returned" in issues
            and len(blockers) == 0
        ):
            blockers.append(
                "Infrastructure blocker detected via preflight: "
                f"{preflight_error}"
            )
            is_blocked = True

        is_ok = (len(issues) == 0) or is_blocked

        return {
            "scenario": name,
            "query": query,
            "ok": is_ok,
            "blocked": is_blocked,
            "issues": issues,
            "blockers": blockers,
            "result": result,
        }

    except Exception as exc:
        print(f"Error: {exc}")
        return {
            "scenario": name,
            "query": query,
            "ok": False,
            "blocked": False,
            "issues": [f"Runtime error: {exc}"],
            "blockers": [],
            "result": None,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual multi-perspective eval")
    parser.add_argument("--quick", action="store_true", help="Run fewer scenarios")
    parser.add_argument(
        "--query",
        default="",
        help="Run a single custom query instead of preset scenarios",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional path to save full JSON report",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.query:
        scenarios = [{"name": "custom", "query": args.query}]
    elif args.quick:
        scenarios = QUICK_SCENARIOS
    else:
        scenarios = DEFAULT_SCENARIOS

    preflight_ok, preflight_error = _weaviate_preflight()
    if preflight_ok:
        print("Weaviate preflight: OK")
    else:
        print("Weaviate preflight: BLOCKED")
        print(f"Reason: {preflight_error}")

    orch = Orchestrator()
    rows = [_run_one(orch, s, preflight_ok, preflight_error) for s in scenarios]

    blocked = [r for r in rows if r.get("blocked", False)]
    failures = [r for r in rows if (not r["ok"]) and (not r.get("blocked", False))]
    passed = [r for r in rows if r["ok"] and (not r.get("blocked", False))]
    print("\n=== Summary ===")
    print(f"Scenarios: {len(rows)}")
    print(f"Passed: {len(passed)}")
    print(f"Blocked (infra): {len(blocked)}")
    print(f"Failed: {len(failures)}")

    report = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "total": len(rows),
        "passed": len(passed),
        "blocked": len(blocked),
        "failed": len(failures),
        "items": rows,
    }

    if args.save:
        target = Path(args.save)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report to: {target}")

    if blocked:
        print("\nNote: blocked scenarios indicate environment/access issues, not model logic bugs.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
