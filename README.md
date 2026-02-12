# EVANA - Emergency Vehicle Assistant for Navigating Accidents

## Background

EVANA is a virtual in-vehicle assistant designed to support drivers and occupants after an accident or vehicle damage. In stressful moments following a collision, people often feel panic and uncertainty about what steps to take next. EVANA provides both emotional support and clear, instructive guidance to help users behave correctly after an accident.

## Architecture

This project implements EVANA as an agentic AI assistant using **LangGraph** with a supervised workflow pattern:

```
                    ┌─────────────────┐
                    │   Supervisor    │
                    │     Agent       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │   Situation     │           │   Empathetic    │
    │     Agent       │           │     Agent       │
    └─────────────────┘           └─────────────────┘
              │
              ▼
    ┌─────────────────┐
    │  Vehicle State  │
    │     (Tool)      │
    └─────────────────┘
```

- **Supervisor**: Coordinates the conversation and delegates to specialized agents
- **Situation Agent**: Retrieves vehicle state (airbags, engine, doors, etc.)
- **Empathetic Agent**: Transforms instructions into calm, supportive messages

## Prerequisites

- [uv](https://docs.astral.sh/uv/) - Python package and project manager
- `OPENAI_API_KEY` environment variable for OpenAI API access (recommended for best output quality)

**Local alternative**: If no API key is set, the project falls back to [Ollama](https://ollama.ai/) with `mistral:7b` model. In this case you must have Ollama installed and the `mistral:7b` model available locally at Ollama's default API url.

## Running the Project

```bash
uv run main.py
```

The assistant will prompt you for input. Use trigger phrases like "help", "emergency", or "Hey EVANA" to start the emergency guidance process. Type `exit` or `quit` to end the conversation.
