"""
Contract tests for the docs-agent MCP server.

WHY these tests exist (response to Chase's question):
  These are not implementation tests. They test the MCP tool *contract* —
  the guarantee that every tool call returns {citation_url, content_text, similarity}.
  That contract holds whether the backend is Groq, vLLM, or any future LLM.
  A broken tool that silently drops citation URLs passes a manual smoke test
  but fails here, catching regressions before users see them.

  See design doc Tests section: "test the contract, not the implementation."

Three test layers (per Kundan's framing):
  1. MCP contract tests — required fields in every tool response
  2. execute_tool dispatch — correct routing, graceful unknown-tool errors
  3. Health check — endpoint returns 200

Run: python -m pytest server/tests/test_app.py -v
No external services required — Milvus is fully mocked.
"""

import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run an async coroutine in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def make_tool_call(name: str, arguments: dict) -> dict:
    """Build a tool_call dict in the shape execute_tool() expects."""
    return {
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        }
    }


def make_hit(citation_url: str, content_text: str, similarity: float = 0.85) -> dict:
    """Build a Milvus search hit dict in the shape MilvusSearchClient.search() returns."""
    return {
        "citation_url": citation_url,
        "content_text": content_text,
        "file_path": "website/docs/example.md",
        "similarity": similarity,
    }


# ---------------------------------------------------------------------------
# 1. MCP Contract Tests
#    These MUST pass regardless of backend (Groq, vLLM, Milvus version).
#    Contract: search_kubeflow_docs returns citation URLs and content text.
# ---------------------------------------------------------------------------

class TestMCPToolContracts(unittest.TestCase):
    """
    The MCP thin-context package must always contain source references.
    If citation_url is missing, the IDE agent cannot route to the source —
    the whole point of the tool is defeated silently.
    """

    def test_search_docs_returns_citation_url(self):
        """Contract: tool result must contain at least one citation URL."""
        hit = make_hit("https://kubeflow.org/docs/components/pipelines/", "Pipeline docs")
        with patch("server.app._milvus_client") as mock_client:
            mock_client.search.return_value = {"results": [hit]}
            tool_call = make_tool_call("search_kubeflow_docs", {"query": "KFP pipeline"})
            result_text, citations = run(__import__("server.app", fromlist=["execute_tool"]).execute_tool(tool_call))
            self.assertTrue(len(citations) > 0, "Contract violated: no citation URLs returned")
            self.assertIn("kubeflow.org", citations[0])

    def test_search_docs_returns_content_text(self):
        """Contract: tool result text must contain content from the search hit."""
        hit = make_hit("https://kubeflow.org/docs/components/kserve/", "KServe content here")
        with patch("server.app._milvus_client") as mock_client:
            mock_client.search.return_value = {"results": [hit]}
            tool_call = make_tool_call("search_kubeflow_docs", {"query": "KServe"})
            result_text, _ = run(__import__("server.app", fromlist=["execute_tool"]).execute_tool(tool_call))
            self.assertIn("KServe content here", result_text)

    def test_search_docs_deduplicates_citations(self):
        """Contract: duplicate citation URLs must not appear in the citations list."""
        url = "https://kubeflow.org/docs/components/pipelines/"
        hits = [make_hit(url, f"chunk {i}") for i in range(3)]
        with patch("server.app._milvus_client") as mock_client:
            mock_client.search.return_value = {"results": hits}
            tool_call = make_tool_call("search_kubeflow_docs", {"query": "pipeline"})
            _, citations = run(__import__("server.app", fromlist=["execute_tool"]).execute_tool(tool_call))
            self.assertEqual(citations.count(url), 1, "Contract violated: duplicate citations returned")

    def test_empty_results_does_not_crash(self):
        """Contract: zero Milvus hits must return a graceful message, not raise."""
        with patch("server.app._milvus_client") as mock_client:
            mock_client.search.return_value = {"results": []}
            tool_call = make_tool_call("search_kubeflow_docs", {"query": "obscure query"})
            result_text, citations = run(__import__("server.app", fromlist=["execute_tool"]).execute_tool(tool_call))
            self.assertIsInstance(result_text, str)
            self.assertEqual(citations, [])

    def test_missing_citation_url_excluded_from_citations(self):
        """Contract: hits with no citation_url must not pollute the citations list."""
        hit = {"citation_url": "", "content_text": "some content", "file_path": "x.md", "similarity": 0.7}
        with patch("server.app._milvus_client") as mock_client:
            mock_client.search.return_value = {"results": [hit]}
            tool_call = make_tool_call("search_kubeflow_docs", {"query": "test"})
            _, citations = run(__import__("server.app", fromlist=["execute_tool"]).execute_tool(tool_call))
            self.assertEqual(citations, [], "Empty citation_url must not appear in citations list")


# ---------------------------------------------------------------------------
# 2. execute_tool Dispatch Tests
# ---------------------------------------------------------------------------

class TestExecuteToolDispatch(unittest.TestCase):

    def test_unknown_tool_returns_error_string(self):
        """Unknown tool names must return an error string, not raise an exception."""
        tool_call = make_tool_call("nonexistent_tool", {})
        result_text, citations = run(__import__("server.app", fromlist=["execute_tool"]).execute_tool(tool_call))
        self.assertIn("Unknown tool", result_text)
        self.assertEqual(citations, [])

    def test_malformed_arguments_handled_gracefully(self):
        """Malformed JSON arguments must not crash the server."""
        tool_call = {
            "function": {
                "name": "search_kubeflow_docs",
                "arguments": "{INVALID JSON",
            }
        }
        # Should return an error string, not propagate JSONDecodeError
        result_text, citations = run(__import__("server.app", fromlist=["execute_tool"]).execute_tool(tool_call))
        self.assertIsInstance(result_text, str)

    def test_top_k_parameter_passed_to_search(self):
        """top_k argument must be forwarded to the search client."""
        with patch("server.app._milvus_client") as mock_client:
            mock_client.search.return_value = {"results": []}
            tool_call = make_tool_call("search_kubeflow_docs", {"query": "KFP", "top_k": 3})
            run(__import__("server.app", fromlist=["execute_tool"]).execute_tool(tool_call))
            mock_client.search.assert_called_once_with("KFP", 3)

    def test_default_top_k_is_five(self):
        """When top_k is omitted, search must be called with default value 5."""
        with patch("server.app._milvus_client") as mock_client:
            mock_client.search.return_value = {"results": []}
            tool_call = make_tool_call("search_kubeflow_docs", {"query": "KFP"})
            run(__import__("server.app", fromlist=["execute_tool"]).execute_tool(tool_call))
            mock_client.search.assert_called_once_with("KFP", 5)


# ---------------------------------------------------------------------------
# 3. Health Check
# ---------------------------------------------------------------------------

class TestHealthCheck(unittest.TestCase):

    def test_health_check_returns_ok(self):
        """Health endpoint must return HTTP 200 for K8s liveness probes."""
        from aiohttp.test_utils import make_mocked_request
        from server.app import health_check

        request = MagicMock()
        response = run(health_check(request))
        self.assertEqual(response.status, 200)


if __name__ == "__main__":
    unittest.main()
