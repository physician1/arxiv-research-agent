"""Local Gradio interface. Start with python app.py."""
import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import tempfile
from pathlib import Path

import gradio as gr

from agent import DEFAULT_MODEL, ModelClient, ModelError, run_agent, source_markdown
from research_tools import ResearchTools, ToolError, bibtex


def research(prompt, provider, model, token, paper_limit):
    try:
        client = ModelClient(provider, model, token)
        for result in run_agent(prompt, client, int(paper_limit)):
            yield result.report, result.activity, result.sources, result.citations
    except (ModelError, ToolError) as exc:
        yield f"**Setup needed:** {exc}", "No research run started.", "No sources retrieved.", ""


def check_connection(provider, model, token):
    try:
        client = ModelClient(provider, model, token)
        result = client.complete([
            {"role": "system", "content": "Test tool calling. Call get_paper with paper_id 1706.03762. Do not answer in text."},
            {"role": "user", "content": "Call the requested tool now."},
        ])
        if any(c["function"]["name"] == "get_paper" for c in result.get("tool_calls", [])):
            return "Connection successful. The model returned a tool call; you can run the agent."
        return "Connected, but the model did not return the requested tool call. Retry or choose another tool-capable model."
    except ModelError as exc:
        return str(exc)


def search_only(topic, count, sort):
    try:
        tools = ResearchTools(int(count))
        result = tools.search_arxiv(topic, int(count), sort)
        return source_markdown(result["papers"]), bibtex(result["papers"])
    except ToolError as exc:
        return str(exc), ""


def export_report(report, sources, citations):
    folder = Path(tempfile.mkdtemp(prefix="arxiv-report-"))
    path = folder / "research-report.md"
    path.write_text(f"{report}\n\n## Retrieved sources\n\n{sources}\n\n## BibTeX\n\n```bibtex\n{citations}\n```\n", encoding="utf-8")
    return str(path)


def export_bibtex(citations):
    folder = Path(tempfile.mkdtemp(prefix="arxiv-bibtex-"))
    path = folder / "references.bib"
    path.write_text(citations or "% No papers retrieved.\n", encoding="utf-8")
    return str(path)


CSS = """
.gradio-container {max-width: 1120px !important; margin: auto;}
#hero {padding: 22px 0 12px;}
#hero h1 {font-size: 2.3rem; letter-spacing: -.04em;}
#hero p {font-size: 1.05rem; color: #667085;}
footer {display: none !important;}
"""


def build_app():
    with gr.Blocks(title="Paper Trail | arXiv Research Agent", analytics_enabled=False,
                   delete_cache=(3600, 86400)) as demo:
        gr.Markdown("# Paper Trail\nSearch papers. Read the evidence. Build a cited research report.", elem_id="hero")
        with gr.Accordion("1 · Connect your AI model", open=True):
            with gr.Row():
                provider = gr.Dropdown(["Hugging Face", "Ollama (local)"], value="Hugging Face", label="AI provider")
                model = gr.Textbox(value=DEFAULT_MODEL, label="Model")
            token = gr.Textbox(type="password", label="Hugging Face token", placeholder="Paste your hf_ token here",
                               info="Used in memory for requests; this app does not save it to a file. Leave blank for Ollama or an existing HF_TOKEN environment variable.")
            gr.Markdown("**First time?** [Create a Hugging Face account](https://huggingface.co/join), then "
                        "[create a token](https://huggingface.co/settings/tokens) with **Make calls to Inference Providers** permission. "
                        "Paste it above and test the connection. Your prompt and retrieved excerpts are sent to the selected AI provider. "
                        "[Inference usage and credits](https://huggingface.co/docs/inference-providers/pricing).")
            test = gr.Button("Test connection", size="sm")
            connection = gr.Textbox(label="Connection status", interactive=False, value="Add a token to enable AI research. Search papers below works without one.")
            provider.change(lambda p: "qwen3:8b" if p == "Ollama (local)" else DEFAULT_MODEL, provider, model, api_name=False)
            test.click(check_connection, [provider, model, token], connection, api_name=False)
        with gr.Tabs():
            with gr.Tab("AI research"):
                prompt = gr.Textbox(label="2 · What would you like to research?", lines=3,
                                    placeholder="Find two recent papers on retrieval-augmented generation. Read their PDFs and summarize their methods and findings.", max_length=2000)
                gr.Examples(examples=[
                    ["Find 2 recent papers on retrieval augmented generation. Read their PDFs and summarize their methods and findings."],
                    ["Read arXiv paper 1706.03762 and explain its method and main findings using supporting passages."],
                    ["Find 2 papers about neural stochastic differential equations. Summarize their abstracts and provide BibTeX."],
                ], inputs=prompt, cache_examples=False)
                with gr.Row():
                    limit = gr.Slider(1, 5, value=2, step=1, label="Maximum papers per search / PDFs per run")
                    run = gr.Button("Research papers", variant="primary")
                report = gr.Markdown("Your research report will appear here.")
                with gr.Accordion("Agent activity", open=True):
                    activity = gr.Textbox(label="Actions and observations", lines=8, interactive=False)
                with gr.Accordion("Retrieved sources", open=False):
                    sources = gr.Markdown("No sources retrieved yet.")
                with gr.Accordion("BibTeX", open=False):
                    citations = gr.Code(language=None, label="References from arXiv metadata", interactive=False)
                with gr.Row():
                    save_report = gr.Button("Prepare report download")
                    save_bib = gr.Button("Prepare BibTeX download")
                download = gr.File(label="Download", interactive=False)
                run.click(research, [prompt, provider, model, token, limit],
                          [report, activity, sources, citations], api_name=False,
                          concurrency_limit=1, trigger_mode="once")
                save_report.click(export_report, [report, sources, citations], download, api_name=False)
                save_bib.click(export_bibtex, citations, download, api_name=False)
            with gr.Tab("Search without AI"):
                gr.Markdown("Search live arXiv metadata and export citations without an AI connection. This tab does not run the agent or generate summaries.")
                topic = gr.Textbox(label="Topic or arXiv search query", placeholder="neural stochastic differential equations")
                with gr.Row():
                    count = gr.Slider(1, 5, value=2, step=1, label="Number of papers")
                    sort = gr.Radio(["relevance", "newest"], value="relevance", label="Sort by")
                search = gr.Button("Search arXiv", variant="primary")
                search_results = gr.Markdown()
                search_bib = gr.Code(language=None, label="BibTeX", interactive=False)
                search.click(search_only, [topic, count, sort], [search_results, search_bib], api_name="search_papers", concurrency_limit=1)
            with gr.Tab("About this lab"):
                gr.Markdown("""## Option 2: improve the arXiv agent
**Goal:** demonstrate a complete AI agent application running locally.

The model selects a search or paper, reads tool results, requests PDF pages, and submits a report. The application checks the source IDs and quoted passages before displaying that report. The activity panel records tool actions and outcomes.

### Improvements
- Search by relevance or newest publication date.
- Retrieve PDFs in bounded page ranges and label the evidence used.
- Attach exact supporting passages to summary claims.
- Generate BibTeX directly from retrieved metadata.
- Handle failed requests, invalid tools, unreadable PDFs, and run limits.
- Export a report and citation file.

### Boundaries
The tools can only retrieve arXiv material. They cannot execute model-written code. Retrieved text is treated as evidence, with instructions to ignore embedded commands. PDF downloads are limited to 15 MB; parsing runs separately with a 25-second timeout. These controls reduce risk but do not provide a complete security sandbox.

Quote matching verifies that a passage exists, not that the model's interpretation is correct. Scanned PDFs need OCR, which is not included. Each research request starts a fresh run; this is not a persistent conversation.
""")
    return demo


if __name__ == "__main__":
    build_app().queue(max_size=8).launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7860")),
                                      share=False, inbrowser=os.getenv("OPEN_BROWSER", "1") == "1",
                                      theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"), css=CSS)
