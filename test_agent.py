"""Offline checks for real failure modes; fake model responses are explicit."""
import io
import json
import unittest
from unittest.mock import patch

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfWriter

from agent import ModelClient, ModelError, run_agent, safe_markdown
from pdf_worker import extract
from research_tools import ResearchTools, ToolError, bibtex, fetch_bytes, normalize_id, web_search

PAPER = {"paper_id": "1706.03762v1", "title": "A test title with {braces} & symbols",
         "authors": ["Example Author"], "published": "2017-06-12",
         "abstract": "This fixture describes a research method using attention to encode input sequences.",
         "url": "https://arxiv.org/abs/1706.03762v1"}
FINDING = {"paper_id": PAPER["paper_id"], "aspect": "Method",
           "text": "The method uses attention to encode input sequences.",
           "quote": PAPER["abstract"], "page": None}


def call(name, args):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.observations = []

    def complete(self, messages):
        self.observations = list(messages)
        return next(self.responses)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tools = ResearchTools(2)
        self.tools.papers[PAPER["paper_id"]] = PAPER.copy()

    def test_identifiers_and_untrusted_urls(self):
        self.assertEqual(normalize_id("https://arxiv.org/pdf/1706.03762v1.pdf"), "1706.03762v1")
        self.assertEqual(normalize_id("hep-th/9901001"), "hep-th/9901001")
        for value in ["https://evil.example/1706.03762", "http://127.0.0.1/secret", "../../secret",
                      "https://arxiv.org@evil.example/abs/1706.03762", "https://arxiv.org/abs/1706.03762?url=foo"]:
            with self.subTest(value=value), self.assertRaises(ToolError):
                normalize_id(value)

    def test_download_host_guard(self):
        with patch("research_tools.httpx.stream") as stream:
            with self.assertRaises(ToolError):
                fetch_bytes("https://example.com/paper.pdf")
            stream.assert_not_called()

    def test_exact_quote_and_real_page_required(self):
        self.tools.verify_findings([FINDING])
        for change in [{"quote": "This is a fabricated quotation that never appeared."},
                       {"paper_id": "9999.12345"}, {"page": 2}]:
            with self.subTest(change=change), self.assertRaises(ToolError):
                self.tools.verify_findings([{**FINDING, **change}])
        self.tools.pages[(PAPER["paper_id"], 2)] = PAPER["abstract"]
        self.tools.verify_findings([{**FINDING, "page": 2}])

    def test_passage_selection_inserts_original_text_and_location(self):
        original = "This PDF uses a ligature: ﬁndings, with a line\nbreak and an en–dash in the extracted passage."
        self.tools.pages[(PAPER["paper_id"], 3)] = original
        passages = self.tools.passages(PAPER["paper_id"], 3, original)
        selection = {"evidence_id": passages[0]["evidence_id"], "aspect": "Summary", "text": "A summary backed by the selected PDF passage."}
        result = self.tools.verify_findings([selection])[0]
        self.assertEqual(result["page"], 3)
        self.assertEqual(result["quote"], " ".join(original.split()))
        self.assertEqual(self.tools.passages(PAPER["paper_id"], 3, original), passages)
        with self.assertRaises(ToolError):
            self.tools.verify_findings([{**selection, "evidence_id": "E99999"}])
        final = list(run_agent("Summarize this paper", ScriptedModel([call("finish_report", {"findings": [selection]})]), research=self.tools))[-1]
        self.assertIn("Report complete", final.activity)
        self.assertIn("PDF page 3", final.report)

    def test_repeated_invalid_reports_stop_early(self):
        invalid = {"evidence_id": "missing", "aspect": "Summary", "text": "A sufficiently long finding."}
        model = ScriptedModel([call("finish_report", {"findings": [invalid]})] * 2)
        final = list(run_agent("Summarize this paper", model, research=self.tools))[-1]
        self.assertIn("Report validation stopped", final.report)

    def test_pdf_bounds_and_unseen_paper(self):
        for args in [{"paper_id": "1706.03762v1", "page_count": 9},
                     {"paper_id": "1706.03762v1", "start_page": 0},
                     {"paper_id": "1706.03762v1", "page_count": True},
                     {"paper_id": "1234.56789"}]:
            with self.subTest(args=args), self.assertRaises(ToolError):
                self.tools.read_paper(**args)

    def test_empty_pdf_and_out_of_range_page(self):
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        buffer = io.BytesIO()
        writer.write(buffer)
        self.assertIn("no extractable text", extract(buffer.getvalue(), 1, 1)["error"])
        self.assertIn("1 pages", extract(buffer.getvalue(), 2, 1)["error"])

    def test_bibtex_comes_from_metadata_and_escapes_symbols(self):
        result = bibtex([PAPER])
        self.assertIn(r"\{braces\} \&", result)
        self.assertIn("year = {2017}", result)
        self.assertIn("eprint = {1706.03762v1}", result)

    def test_model_can_recover_from_bad_quote(self):
        model = ScriptedModel([
            call("finish_report", {"findings": [{**FINDING, "quote": "An invented quote that cannot match the source text."}]}),
            call("finish_report", {"findings": [FINDING]}),
        ])
        final = list(run_agent("Summarize this paper", model, research=self.tools))[-1]
        self.assertIn("Report complete", final.activity)
        self.assertIn("does not match", final.activity)
        self.assertIn("abstract only", final.report)
        self.assertTrue(any(m["role"] == "tool" and "error" in m["content"] for m in model.observations))

    def test_unknown_tool_is_rejected_then_agent_can_recover(self):
        model = ScriptedModel([call("run_shell", {"command": "echo unsafe"}),
                               call("finish_report", {"findings": [FINDING]})])
        final = list(run_agent("Summarize the paper", model, research=self.tools))[-1]
        self.assertIn("not allowed", final.activity)
        self.assertIn("Report complete", final.activity)

    def test_endless_loop_stops(self):
        model = ScriptedModel([call("unknown_tool", {})] * 10)
        final = list(run_agent("Search for papers", model))[-1]
        self.assertIn("Run limit reached", final.report)

    def test_search_read_finish_loop(self):
        tools = ResearchTools(2)
        model = ScriptedModel([
            call("search_arxiv", {"query": "attention", "max_results": 1, "sort": "newest"}),
            call("read_paper", {"paper_id": PAPER["paper_id"]}),
            call("finish_report", {"findings": [{**FINDING, "page": 1}]})])
        def fake_read(**kwargs):
            tools.pages[(PAPER["paper_id"], 1)] = PAPER["abstract"]
            return {"pages": [{"page": 1, "text": PAPER["abstract"]}], "total_pages": 1}
        with patch("research_tools.api_query", return_value=[PAPER]) as api, patch.object(tools, "read_paper", side_effect=fake_read):
            final = list(run_agent("Find recent attention papers and read the PDF", model, research=tools))[-1]
        self.assertEqual(api.call_args.args[0]["sortBy"], "submittedDate")
        self.assertIn("PDF page 1", final.report)
        self.assertIn("@misc", final.citations)

    def test_auth_error_does_not_reveal_token(self):
        response = httpx.Response(401, request=httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions"))
        with patch("agent.httpx.Client") as client:
            client.return_value.__enter__.return_value.post.return_value = response
            with self.assertRaises(ModelError) as caught:
                ModelClient(token="hf_TEST_SECRET").complete([])
        self.assertNotIn("hf_TEST_SECRET", str(caught.exception))

    def test_permission_error_retains_reason_and_redacts_credentials(self):
        response = httpx.Response(403, json={"error": "Inference permission missing for hf_TESTSECRET"},
                                  request=httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions"))
        with patch("agent.httpx.Client") as client:
            client.return_value.__enter__.return_value.post.return_value = response
            with self.assertRaises(ModelError) as caught:
                ModelClient(token="hf_TESTSECRET").complete([])
        self.assertIn("HTTP 403", str(caught.exception))
        self.assertIn("Inference permission missing", str(caught.exception))
        self.assertNotIn("hf_TESTSECRET", str(caught.exception))

    def test_source_markup_is_escaped(self):
        text = safe_markdown('<script>alert(1)</script> ![x](https://evil.example)')
        self.assertNotIn("<script>", text)
        self.assertIn(r"\!\[", text)

    def test_official_site_fallback_preserves_metadata(self):
        fixture = '''<li class="arxiv-result"><p class="list-title"><a href="https://arxiv.org/abs/1706.03762">paper</a></p>
        <p class="title">Attention Is All You Need</p><p class="authors"><a>Example Author</a></p>
        <span class="abstract-full" id="1706.03762v7-abstract-full">A real abstract in the test fixture.<a>Less</a></span>
        <p>Submitted 12 June, 2017;</p></li>'''
        with patch("research_tools.web_fetch", return_value=BeautifulSoup(fixture, "html.parser")):
            papers = web_search("ti:attention", 1, "newest")
        self.assertEqual(papers[0]["paper_id"], "1706.03762v7")
        self.assertEqual(papers[0]["published"], "2017-06-12")
        self.assertNotIn("Less", papers[0]["abstract"])

    def test_api_failure_uses_site_fallback(self):
        with patch("research_tools.api_query", side_effect=ToolError("HTTP 406")), \
                patch("research_tools.web_search", return_value=[PAPER]) as fallback:
            result = self.tools.search_arxiv("attention", 1, "newest")
        fallback.assert_called_once_with("attention", 1, "newest")
        self.assertIn("website fallback", result["retrieval"])

    def test_external_redirect_blocked_before_second_request(self):
        redirect = httpx.Response(302, headers={"Location": "https://evil.example/paper.pdf"},
                                 request=httpx.Request("GET", "https://arxiv.org/pdf/1706.03762"))
        with patch("research_tools.httpx.stream") as stream:
            stream.return_value.__enter__.return_value = redirect
            with self.assertRaises(ToolError):
                fetch_bytes("https://arxiv.org/pdf/1706.03762")
            self.assertEqual(stream.call_count, 1)

    def test_download_size_limit(self):
        response = httpx.Response(200, content=b"A" * 100,
                                  request=httpx.Request("GET", "https://arxiv.org/pdf/1706.03762"))
        with patch("research_tools.httpx.stream") as stream:
            stream.return_value.__enter__.return_value = response
            with self.assertRaises(ToolError):
                fetch_bytes("https://arxiv.org/pdf/1706.03762", limit=50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
