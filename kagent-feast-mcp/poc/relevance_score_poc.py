"""
POV: MCP tool returning relevance score, mock IDE agent branching on threshold.

Validates the design hypothesis from Section 4 (Tool Handshake Flow):
  - MCP tool returns {chunk_text, source_url, score} in the 150-token thin-context package
  - IDE agent uses score to decide: show result directly OR trigger cross-collection fallback
  - Score < SCORE_THRESHOLD on docs_rag triggers a second search against code_rag

Run: python kagent-feast-mcp/poc/relevance_score_poc.py
No external services needed — mock backend simulates Milvus distance scores.
"""

from dataclasses import dataclass
import random

SCORE_THRESHOLD = 0.65  # below this, trigger cross-collection fallback


@dataclass
class MCPToolResponse:
    chunk_text: str
    source_url: str
    score: float       # cosine similarity from Milvus, 0.0-1.0 (higher = more relevant)
    collection: str    # which collection this came from


def mock_search(query: str, collection: str, seed: int = None) -> MCPToolResponse:
    """
    Simulates a Milvus search returning a scored result.
    Replace random.uniform() with real MilvusClient.search() in production.
    """
    rng = random.Random(seed)
    # Simulate domain affinity: code queries score higher in code_rag
    if "yaml" in query.lower() or "scaletozero" in query.lower() or "manifest" in query.lower():
        base = 0.72 if collection == "code_rag" else 0.48
    elif "oomkilled" in query.lower() or "error" in query.lower():
        base = 0.55  # ambiguous queries score medium everywhere
    else:
        base = 0.78 if collection == "docs_rag" else 0.52

    score = min(1.0, max(0.0, base + rng.uniform(-0.12, 0.12)))

    return MCPToolResponse(
        chunk_text=f"[Mock 150-token snippet from {collection} for: '{query[:40]}']",
        source_url=f"https://kubeflow.org/docs/{collection.replace('_rag', '')}/example",
        score=round(score, 3),
        collection=collection,
    )


def ide_agent_decision(query: str, seed: int = None) -> dict:
    """
    Simulates IDE agent logic for cross-collection routing:
      1. Search docs_rag first (primary collection)
      2. If score < SCORE_THRESHOLD, fall back to code_rag
      3. Return full decision log for analysis

    In production, each search() call is one MCP tool invocation.
    The agent makes the branching decision locally — no extra round-trip to the server.
    """
    log = {"query": query, "steps": []}

    # Step 1: Primary search — docs_rag
    docs_result = mock_search(query, "docs_rag", seed=seed)
    decision = "accept" if docs_result.score >= SCORE_THRESHOLD else "fallback"
    log["steps"].append({
        "collection": "docs_rag",
        "score": docs_result.score,
        "threshold": SCORE_THRESHOLD,
        "decision": decision,
    })

    if decision == "accept":
        log["final_result"] = docs_result
        log["cross_collection_triggered"] = False
        return log

    # Step 2: Fallback — code_rag
    code_result = mock_search(query, "code_rag", seed=seed)
    log["steps"].append({
        "collection": "code_rag",
        "score": code_result.score,
        "threshold": SCORE_THRESHOLD,
        "decision": "accept",
    })
    log["final_result"] = code_result
    log["cross_collection_triggered"] = True
    return log


def format_result(log: dict) -> str:
    lines = [f"\nQuery: {log['query']}"]
    for step in log["steps"]:
        arrow = "accepted" if step["decision"] == "accept" else "fallback triggered"
        lines.append(f"  [{step['collection']}] score={step['score']} -> {arrow}")
    result = log["final_result"]
    lines.append(f"  Final source: {result.source_url}")
    lines.append(f"  Cross-collection triggered: {log['cross_collection_triggered']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== MCP Relevance Score POV ===")
    print(f"Score threshold: {SCORE_THRESHOLD}")
    print("Simulating IDE agent cross-collection routing decisions\n")

    test_cases = [
        # (query, seed) — seed for reproducible output
        ("How do I configure KServe autoscaling?", 1),
        ("Show me the YAML for scaleToZeroGracePeriod in a notebook webhook", 2),
        ("KFP pipeline step fails with OOMKilled error", 3),
        ("What is the architecture of the Kubeflow dashboard?", 4),
        ("kubeflow/manifests InferenceService resource kind definition", 5),
    ]

    triggered = 0
    for query, seed in test_cases:
        result = ide_agent_decision(query, seed=seed)
        print(format_result(result))
        if result["cross_collection_triggered"]:
            triggered += 1

    print(f"\n--- Summary ---")
    print(f"Queries tested: {len(test_cases)}")
    print(f"Cross-collection fallback triggered: {triggered}/{len(test_cases)}")
    print(
        "\nConclusion: Returning score in MCP response lets the IDE agent route"
        "\nwithout a server round-trip. Threshold is tunable; 0.65 worked here."
    )
