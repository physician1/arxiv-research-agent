# Validation status

Test environment: Python 3.12 on Windows.

## Verified

- All 19 offline tests pass, covering source verification, passage selection, page restrictions, unsafe URLs and redirects, download limits, empty PDFs, unknown tools, bounded retries, error recovery, BibTeX, and the official arXiv website fallback.
- Simulated model responses exercise the search, PDF reading, and report loop. These verify orchestration, not real model output quality.
- The Gradio interface runs locally and supports report and BibTeX exports.
- Live arXiv search returned metadata and abstracts in the browser.
- Live retrieval of arXiv paper 1706.03762 returned Attention Is All You Need, version 7. Its PDF contains 15 pages.
- The user confirmed successful Hugging Face authentication and supplied a live agent trace showing metadata retrieval followed by extraction of PDF pages 1 through 4.

## Report-generation limitation

The first live run repeatedly failed exact-quote validation, then stopped when the inference service reported exhausted credits. No completed live report was produced in that run.

The revised report tool asks the model to select existing evidence IDs. The application inserts the original passages and source locations, avoiding the need for the model to retype PDF text. Offline tests cover this change, including ligatures and line breaks.

A complete live run with the revised report method remains unverified because inference credits were exhausted. A local Ollama adapter is included but has not been tested with a live local model.

## Run limits and evidence interpretation

Runs allow at most six model rounds and ten tool calls, and stop after two invalid report attempts. Passage checks establish that the evidence exists; they do not establish that the model's interpretation is correct.

See README.md for setup and further limitations.
