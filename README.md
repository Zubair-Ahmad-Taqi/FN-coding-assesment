# Clinic Appointment Agent

A small, local-first AI appointment assistant built for the Futurenostics hands-on assessment. It supports safe appointment booking, Ollama tool calling, grounded clinic Q&A, async patient lookups, and workflow tracing.

## Features

- Safe `book_appointment()` tool with input validation, availability checks, duplicate prevention, API timeout/failure handling, and consistent responses.
- Local Ollama integration for tool selection and RAG answers.
- Interactive CLI chat with multi-turn booking memory.
- Keyword/IDF RAG over the supplied clinic knowledge base; answers include source chunks and decline unsupported questions.
- Async patient client using `httpx`, with timeout, non-200, and network-failure handling plus concurrent batch requests.
- Trace logs containing a trace ID, tool name, tool arguments, result, and latency.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- The `qwen2.5:3b` model (or another tool-capable local model)

## Setup

```powershell
cd "C:\Users\Admin\Desktop\Futurenostics Coding Assessment"
pip install -r requirements.txt
ollama pull qwen2.5:3b
```

Start Ollama in a separate terminal if it is not already running:

```powershell
ollama serve
```

Optionally select a different installed model:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:3b"
```

## Run the CLI chatbot

```powershell
python agent.py
```

Example conversation:

```text
You: What days is Dr. Ahmed available?
You: Book my appointment with Dr. Ahmed on Wednesday at 9 AM.
Assistant: Please provide: patient_id. Use YYYY-MM-DD for a calendar date.

You: My patient ID is PAT-123.
```

Booking details are remembered during the CLI session. Dates must use `YYYY-MM-DD` when a specific calendar date is given, avoiding ambiguity such as `02/10/2026`.

Show the raw agent/tool payload separately for debugging:

```powershell
python agent.py --show-tool-response
```

Exit the CLI with `quit` or `exit`.

## Run the API (optional)

```powershell
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for API documentation.

| Route | Purpose |
|---|---|
| `POST /appointments` | Direct safe appointment booking |
| `POST /agent/requests` | Structured booking-controller request |
| `POST /knowledge/answer` | Grounded clinic knowledge-base answer |

## Run tests

```powershell
python -m unittest -v
```

## Project structure

| File | Responsibility |
|---|---|
| `appointment_booking.py` | Booking domain tool and mocked external services |
| `agent.py` | Ollama tool schema, controller, tracing, conversation memory, and CLI |
| `rag.py` | Knowledge base, retrieval ranking, and grounded Ollama answers |
| `ollama_client.py` | Dependency-free local Ollama client |
| `patient_service.py` | Async single and concurrent patient fetching |
| `main.py` | Optional FastAPI interface |

## Trace logging

When a booking tool is invoked, the CLI logs information similar to:

```text
trace_id=<uuid> tool=book_appointment tool_arguments={...}
trace_id=<uuid> tool=book_appointment tool_result={...} latency_ms=<number>
```

The booking is only reported as successful when `book_appointment()` returns a successful backend result.
