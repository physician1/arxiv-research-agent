# Lab demonstration guide

## Before recording or presenting

Start the app and connect a tool-capable model. Run the tests. Keep the token field collapsed during the demonstration. Complete at least one live AI run before describing the assignment as finished.

## 1. Explain the project

“I chose option 2 and extended the arXiv research agent. My application runs locally. It can search for papers, read PDF pages, and produce a summary with supporting passages and BibTeX citations.”

Use your own words and be ready to explain the files and design choices.

## 2. Demonstrate a specific paper

Paste:

> Read arXiv paper 1706.03762 and explain its method and main findings using supporting passages.

Show the activity panel. Look for metadata retrieval, PDF reading, and successful report validation. Open a paper link and compare one supporting quote against the cited PDF page. Download the report and BibTeX.

The actual action sequence is chosen by the model, so do not claim a particular sequence before observing it.

## 3. Demonstrate search

Paste:

> Find 2 recent papers on retrieval augmented generation. Read their PDFs and summarize their methods and findings.

Show that the agent searches, reads results, and selects papers. Verify that the search uses newest ordering when requested. Search results can vary over time.

## 4. Explain the controls

- Tools only retrieve arXiv material; arbitrary execution and unrelated URLs are blocked.
- Every reported quote must match text the application actually retrieved.
- Citation metadata is assembled by the application.
- Failed tool calls become observations the model can use to recover.
- Page, download, and step limits keep each run bounded.

Run `python -m unittest -v test_agent` to demonstrate automated checks for fabricated quotes, unknown tools, invalid URLs, and retry limits. These tests use scripted model responses; they are not evidence of live model performance.

## 5. Explain why this is an agent

The model chooses the next tool and receives the environment's response before deciding what to do next. Searching, reading, correcting invalid arguments, and finishing the report occur in a loop. The “Search without AI” tab is only a supporting utility.

## Suggested evidence to save

- A screenshot of a completed AI report and its activity panel.
- An exported Markdown report and `.bib` file.
- Test output.
- A brief description of improvements and limitations.

The assignment text does not specify a submission format. These are suggested demonstration materials, not additional course requirements.
