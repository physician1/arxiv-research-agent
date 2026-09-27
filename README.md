# Paper Trail: arXiv Research Agent

**Lab approach:** Option 2, improve the provided arXiv agent.

**Validation status:** Live model connectivity, paper retrieval, and PDF reading were demonstrated. Report generation initially failed quote validation and inference credits ran out. The revised passage-selection method passes offline tests, but a completed live report with this revision remains unverified. See [VALIDATION.md](VALIDATION.md).

A local Gradio application in which an AI model chooses tools to search arXiv, retrieve paper metadata, read PDF pages, and build a report with supporting passages. No cloud hosting is required. The default AI connection uses Hugging Face inference; Ollama is also supported for local inference.

## Start on this computer

Double-click **START.cmd**. It opens the app at **http://127.0.0.1:7860**. Keep the terminal window open while using it. Close the terminal or press Ctrl+C there to stop it.

The prepared Python environment is reused on this computer. On another Windows computer, install Python 3.10 or later and run START.cmd; the launcher creates an environment and installs the requirements. Internet is needed for arXiv and Hugging Face.

On macOS/Linux, or for manual setup:

```sh
python -m venv .venv
# Activate .venv with the command appropriate for your operating system.
python -m pip install -r requirements.txt
python app.py
```

If port 7860 is occupied, close an earlier copy of the app or set `PORT` to another port before launching.

## Connect the AI model

1. Create or sign in to your [Hugging Face account](https://huggingface.co/join).
2. Open [Access Tokens](https://huggingface.co/settings/tokens) and create a fine-grained token for this lab. Enable **Make calls to Inference Providers**. Repository write access is not needed.
3. Open the app's **Connect your AI model** panel. Keep **Hugging Face** selected and paste your token into the password field. This app does not save the token to disk or include it in reports.
4. Click **Test connection**. It checks whether the selected model can return a tool call.
5. Enter a research request and click **Research papers**.

The default model is `Qwen/Qwen3-Coder-30B-A3B-Instruct`. The field is editable because hosted model availability changes. Hugging Face supplies limited monthly inference credits; review [pricing and credits](https://huggingface.co/docs/inference-providers/pricing) before additional usage. Each research run is limited to 6 model rounds and 10 tool calls.

The app can also use an `HF_TOKEN` environment variable. Do not put a real token in source files, screenshots, or a submitted ZIP.

### Optional local model

If Ollama is installed and running, download a model that supports tools (for example `ollama pull qwen3:8b`), choose **Ollama (local)**, and set the installed model name. The app connects only to `http://127.0.0.1:11434`. Local inference speed and memory requirements depend on the model and computer. This option was implemented but not tested with a live local model in this environment.

## Features

- **Autonomous loop:** the model selects tools, sees results/errors, and chooses the next action.
- **Search:** relevance or newest ordering; up to five results per search.
- **Reliable retrieval:** use arXiv's API, with the official website as a fallback. The activity log identifies fallback searches. API newest ordering uses submission date; website newest ordering uses first announcement date.
- **PDF reading:** up to four pages per tool call, 6,500 characters per page, and a user-selected limit of one to five distinct PDFs per run.
- **Evidence:** the model selects a numbered passage for each finding. The app inserts the original passage and its source location, then verifies it against the retrieved text. Two invalid report attempts stop the run.
- **BibTeX:** built from retrieved metadata, without asking the model to invent citation fields.
- **Exports:** Markdown report and `.bib` file.
- **Search without AI:** retrieve live abstracts and citations before configuring a model. This is a utility, not an autonomous agent run.

## How the agent works

```text
User request
    -> model selects a tool
    -> validate tool name and arguments
    -> execute arXiv search / metadata retrieval / PDF reading
    -> send observations or errors back to the model
    -> model selects another action or submits finish_report
    -> verify source IDs and supporting quotes
    -> render report and generate citations
```

`agent.py` implements the loop directly so its decisions and controls are easy to explain. This follows the lab's agent concepts without requiring the reference code or smolagents. The model chooses actions; the application only dispatches an explicit set of tools. Each new request starts with fresh source and document state.

### Files

| File | Purpose |
| --- | --- |
| `app.py` | Gradio interface, connection test, downloads |
| `agent.py` | Model adapters, tool schemas, agent loop, report rendering |
| `research_tools.py` | arXiv retrieval, input checks, evidence matching, BibTeX |
| `pdf_worker.py` | PDF extraction in a separate process |
| `test_agent.py` | Offline regression tests with explicitly simulated model responses |
| `DEMO_GUIDE.md` | Suggested demonstration and explanation |
| `VALIDATION.md` | What has and has not been verified |

## Guardrails and limitations

- Only the four registered research/report tools can be called. No model-generated shell or Python code is executed.
- Paper identifiers and download hosts are validated, including redirects. Downloads are capped at 15 MB; PDF extraction has a 25-second process timeout. This is not a full OS security sandbox or memory quota.
- Retrieved text is marked as untrusted data in the model instructions. Prompt injection cannot be ruled out, but tools cannot send messages, execute arbitrary code, or access arbitrary URLs/files.
- Source IDs and quote existence are checked programmatically. These checks **do not prove that a paraphrase is correct or logically supported**. Review the displayed evidence.
- PDF text can omit tables, figures, mathematics, and scanned content. No OCR is included. Reports identify abstract-only evidence and physical PDF page numbers.
- Metadata and model services can be unavailable. Failures appear as tool observations so the model can revise its request. Bounded runs prevent endless retry loops.
- Only excerpts are passed to the model. Long papers may need several focused requests.
- Tokens stay in the app's/browser's memory for the active page; no persistent credential store is implemented. In Hugging Face mode, prompts and retrieved passages are sent to that inference service. Exported reports contain research content, not credentials.
- The local server binds to 127.0.0.1 and does not create a public share link.

## Tests

```sh
python -m unittest -v test_agent
```

See `VALIDATION.md` for live checks and the remaining credential-dependent test.

## References

- Course handout: *Agentic AI: Build an Autonomous Research Assistant*, Wei Ding and Inal Mashukov, UMass Boston, Fall 2026. The implementation is an independent extension of its arXiv-agent concept.
- [Hugging Face function calling](https://huggingface.co/docs/inference-providers/guides/function-calling)
- [Hugging Face tokens](https://huggingface.co/docs/hub/security-tokens)
- [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html)
- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)
- [Gradio Blocks](https://www.gradio.app/guides/blocks-and-event-listeners)

