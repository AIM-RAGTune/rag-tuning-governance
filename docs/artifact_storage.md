# Artifact Storage

RAGTune supports a small artifact-sink abstraction so the governance engine can write local artifacts and optionally hand them to cloud storage.

Supported modes:

```text
local
azure_blob
s3
gcs
disabled
```

Environment variables:

```text
RAGTUNE_STORAGE_MODE=local|azure_blob|s3|gcs|disabled
RAGTUNE_OUTPUT_ROOT=/outputs
RAGTUNE_AZURE_BLOB_CONTAINER=<container>
RAGTUNE_AZURE_BLOB_PREFIX=<prefix>
RAGTUNE_S3_BUCKET=<bucket>
RAGTUNE_S3_PREFIX=<prefix>
RAGTUNE_GCS_BUCKET=<bucket>
RAGTUNE_GCS_PREFIX=<prefix>
```

`local` mode works without cloud SDKs or credentials. Cloud modes fail closed if optional SDKs or required configuration are unavailable. RAGTune writes a local copy of `promotion_decision.json` before any optional upload attempt.

Storage sinks must not log secrets. Public artifacts should contain only metrics, hashes, IDs, sanitized summaries, and claim-boundary metadata.
