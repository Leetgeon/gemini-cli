# Claude models on Vertex AI

Gemini CLI supports using Claude models hosted on
[Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude).
This lets you use Claude models such as `claude-opus-4-6` directly through the
Gemini CLI interface with your existing Vertex AI authentication.

## Prerequisites

Before using Claude models with Gemini CLI, ensure you have the following:

1. A **Google Cloud project** with billing enabled.
2. The **Vertex AI API** enabled on your project.
3. **Claude model access** enabled in your project through the
   [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden).
4. **Application Default Credentials (ADC)** configured via the
   [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).

## Setup

### 1. Configure Application Default Credentials

Log in to Google Cloud and set up ADC:

```bash
gcloud auth application-default login
```

### 2. Set environment variables

Set the required environment variables for your project and region.

**macOS/Linux**

```bash
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="us-east5"
export GOOGLE_GENAI_USE_VERTEXAI="true"
```

**Windows (PowerShell)**

```powershell
$env:GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
$env:GOOGLE_CLOUD_LOCATION="us-east5"
$env:GOOGLE_GENAI_USE_VERTEXAI="true"
```

> **Note:** Replace `YOUR_PROJECT_ID` with your actual Google Cloud project ID.
> Choose a location where Claude models are available (see
> [Available regions](#available-regions) below).

To make these settings persistent, see
[Persisting Environment Variables](../get-started/authentication.md#persisting-vars).

### 3. Start Gemini CLI with a Claude model

Use the `--model` flag to specify a Claude model:

```bash
gemini --model claude-opus-4-6
```

When prompted, select **Vertex AI** as the authentication method.

## Supported models

You can use any Claude model available on Vertex AI. Specify the model name with
the `--model` flag. For example:

- `claude-opus-4-6`
- `claude-sonnet-4-6`
- `claude-haiku-4-5`

> **Note:** Model availability depends on your region and project configuration.
> Check the
> [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden)
> for the latest available models.

## Supported features

Claude models on Vertex AI support the following Gemini CLI features:

- **Streaming responses** — Real-time streaming of model output.
- **Tool use (function calling)** — Claude can invoke tools defined by Gemini
  CLI to perform actions such as reading files, running shell commands, and
  searching code.
- **Thinking (extended thinking)** — Claude's thinking process is surfaced in
  the CLI, similar to Gemini's thinking mode.

## Limitations

- **Token counting** — Claude does not provide a token counting endpoint on
  Vertex AI. Token counts in usage reports may show as zero.
- **Embeddings** — Claude models do not support embeddings. Use a Gemini
  embedding model (such as `gemini-embedding-001`) for embedding tasks.
- **Model routing** — Automatic model fallback does not apply to Claude models.
  If the Claude model is unavailable, you will receive an error.
- **Sub-agents** — The `--model` flag only sets the primary model. Sub-agents
  still use Gemini models.

## Available regions <a id="available-regions"></a>

Claude models on Vertex AI are available in select regions. Common regions
include:

- `us-east5`
- `europe-west1`

For the most up-to-date list of supported regions, see the
[Claude on Vertex AI documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#regions).

## Troubleshooting

- **Authentication errors:** Ensure ADC is configured
  (`gcloud auth application-default login`) and that `GOOGLE_CLOUD_PROJECT` and
  `GOOGLE_CLOUD_LOCATION` are set correctly.
- **Model not found:** Verify that Claude model access is enabled in your
  project through the Vertex AI Model Garden.
- **Region errors:** Confirm that Claude models are available in the region
  specified by `GOOGLE_CLOUD_LOCATION`.

## What's next?

- [Authentication setup](../get-started/authentication.md) — Configure
  authentication for Gemini CLI.
- [Model selection](./model.md) — Learn about model selection options.
- [Model routing](./model-routing.md) — Learn about automatic model fallback.
