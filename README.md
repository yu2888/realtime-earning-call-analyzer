# Real-Time Earnings Call Analyzer

This project analyzes earnings calls in real time. It captures audio playing
through the default speaker, transcribes it with NVIDIA's Nemotron streaming
speech recognition model, and sends the ASR transcript to an OpenAI-compatible
LLM for analysis. The LLM response streams in a second VS Code terminal while
the live transcript continues in the original terminal.

## Setup

Install [uv](https://docs.astral.sh/uv/), then install the project dependencies:

```powershell
uv sync
```

## Demo

https://github.com/user-attachments/assets/8c6e3ad5-ecc8-48f7-aefc-4c11f464de04





## Run

Start the analyzer, then play the earnings call through the default speaker:

```powershell
uv run stream_system_audio.py
```

Press `Ctrl+C` to stop analysis.

Set the OpenAI-compatible endpoint and model in `.env`:

```dotenv
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_MODEL=local-model
LLM_API_KEY=
```
