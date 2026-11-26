# Precision CRAG Agent

I built a question-answering app that checks whether its search results help answer the question before it writes a response. It demonstrates **Corrective Retrieval-Augmented Generation (CRAG)** on four short notes about FastAPI, LangGraph, CRAG, and AI evaluation.

[Open the demo](https://precision-crag-agent.onrender.com) · [Explore the API](https://precision-crag-agent.onrender.com/docs)

## Try two questions

| Ask                             | What to look for                                                                                                                                                                                         |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What is LangGraph used for?** | The knowledge base contains a note about modeling agent workflows as graphs. The app can use that passage to answer.                                                                                     |
| **Where is Dwaraka?**           | The notes contain no geographic information. When every passage is rejected, the app retries and eventually replies, “I couldn't find enough information in the knowledge base to answer that question.” |

The UI shows the **final search query** and **how many passages passed the relevance check**. These make it easier to inspect an unexpected answer.

## How a question moves through the app

1. **Build the search index.** At startup, FastEmbed converts each note into a numerical representation using `BAAI/bge-small-en-v1.5`. FAISS indexes those vectors in memory.
2. **Find candidates.** FastAPI receives the question. FAISS retrieves the two passages closest in meaning to the search query.
3. **Check each passage.** A Groq-hosted language model judges whether each passage helps answer the user's **original question**. LangGraph routes the request based on the results.
4. **Answer or retry.** If any passage passes, the model receives the retained text and the original question to produce an answer. If none pass, the app rewrites the search query and tries again. After two rewrites (three searches total), an empty result returns the fixed fallback without another model call.

### A bug that shaped the design

During testing, “Where is Dwaraka?” produced an answer about CRAG and LangGraph. The app had accepted two unrelated passages. Another question produced an answer even though zero passages survived. I separated the original question from the rewritten search query, tightened passage grading, and added the explicit empty-result fallback. Rewriting can now change what FAISS searches for while preserving the question used to grade evidence and write the answer.

## Run locally

You need Git, Python 3.11, and a Groq API key. Run these commands in PowerShell:

```powershell
git clone https://github.com/SreelathaVS/precision-crag-agent.git
cd precision-crag-agent
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt langchain-groq fastembed python-dotenv langchain-text-splitters
```

Create a file named `.env` in the repository root with your own key:

```dotenv
GROQ_API_KEY=your_key_here
```

Start the server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.main:app --reload
```

The first startup downloads the embedding model. Once the terminal reports that startup is complete, open the [local UI](http://127.0.0.1:8000/) or [interactive API docs](http://127.0.0.1:8000/docs).

**Docker option.** With Docker running and the same `.env` file in the repository root:

```sh
docker build -t precision-crag-agent .
docker run --rm --env-file .env -p 8000:8000 precision-crag-agent
```

## API

Send the following JSON to `POST /api/v1/query`:

```json
{ "question": "What is LangGraph used for?" }
```

The response includes `answer`, `retrieved_chunks` (the number kept after grading), and `query_used` (the final search wording). The `GET /health` endpoint checks that the web server responds.

## Evaluation

Two regression cases cover LangGraph's purpose and how CRAG handles irrelevant retrieval. Each case requires at least one retained passage. A model judge then compares the generated answer with a reference, allowing paraphrases and compatible details. The suite checks agreement with those two references. The out-of-scope fallback was checked separately through the UI.

```powershell
.\.venv\Scripts\python.exe -m pytest evals/test_evals.py -v
```

These tests call Groq and require a valid key. GitHub Actions runs them on pushes to `main` and pull requests targeting `main`, using the `GROQ_API_KEY` repository secret. [View CI runs](https://github.com/SreelathaVS/precision-crag-agent/actions)

## Where to read the code

| File                                       | What it contains                                                |
| ------------------------------------------ | --------------------------------------------------------------- |
| [src/vector_store.py](src/vector_store.py) | The four notes, embedding model, and FAISS retriever            |
| [src/agent.py](src/agent.py)               | The LangGraph workflow, relevance check, rewrites, and fallback |
| [src/main.py](src/main.py)                 | FastAPI routes and request/response fields                      |
| [static/index.html](static/index.html)     | The browser interface                                           |
| [evals/](evals/)                           | The pytest evaluation and its reference questions and answers   |
