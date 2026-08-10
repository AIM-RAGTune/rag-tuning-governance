# Generator Configuration

Generative validation can use a local or hosted generator. Raw prompts and raw generated answers must remain local-only under `.local_data/`.

## Local Ollama

```bash
export RAGTUNE_GENERATOR_PROVIDER=ollama
export RAGTUNE_OLLAMA_BASE_URL=http://localhost:11434
export RAGTUNE_GENERATOR_MODEL=qwen3:8b
```

Run:

```bash
python3 scripts/run_hotpotqa_generative_llm_validation.py \
  --config configs/experiments/ragtune_hotpotqa_generative_llm_validation_v1.yaml \
  --output-root artifacts/generative_llm_validation/hotpotqa \
  --force
```

## Local OpenAI-Compatible Runtime

```bash
export RAGTUNE_GENERATOR_PROVIDER=local_openai_compatible
export RAGTUNE_LOCAL_OPENAI_BASE_URL=http://localhost:11434/v1
export RAGTUNE_LOCAL_OPENAI_API_KEY=local-not-secret
export RAGTUNE_GENERATOR_MODEL=<local-model-name>
```

## Hosted Models

Hosted validation is optional and requires normal provider credentials in the local environment. Do not commit credentials.

```bash
export RAGTUNE_GENERATOR_PROVIDER=openai
export RAGTUNE_GENERATOR_MODEL=gpt-4o-mini
```

For Azure OpenAI:

```bash
export RAGTUNE_GENERATOR_PROVIDER=azure_openai
export AZURE_OPENAI_DEPLOYMENT=<deployment-name>
export AZURE_OPENAI_API_VERSION=2024-10-21
```

Provider secrets are intentionally omitted from this document.
