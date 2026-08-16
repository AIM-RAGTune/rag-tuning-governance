#!/usr/bin/env bash
set -euo pipefail

report_root="${1:-/tmp/ragtune-k8s-kind-validation}"
/bin/mkdir -p "$report_root"
report_json="$report_root/k8s_kind_validation_report.json"
report_md="$report_root/k8s_kind_validation_report.md"

if ! command -v kubectl >/dev/null 2>&1; then
  printf '{"result_class":"FALLBACK_KUBECTL_UNAVAILABLE","real_cloud_deployment_performed":false}\n' > "$report_json"
  printf '# Kubernetes Validation\n\nResult: `FALLBACK_KUBECTL_UNAVAILABLE`\n' > "$report_md"
  exit 0
fi

if ! command -v kind >/dev/null 2>&1; then
  printf '{"result_class":"FALLBACK_KIND_UNAVAILABLE","real_cloud_deployment_performed":false}\n' > "$report_json"
  printf '# Kubernetes Validation\n\nResult: `FALLBACK_KIND_UNAVAILABLE`\n' > "$report_md"
  exit 0
fi

kubectl apply --dry-run=client -k deploy/kubernetes > "$report_root/kubectl_dry_run.txt"
printf '{"result_class":"K8S_KIND_STATIC_DRY_RUN_PASSED","real_cloud_deployment_performed":false}\n' > "$report_json"
printf '# Kubernetes Validation\n\nResult: `K8S_KIND_STATIC_DRY_RUN_PASSED`\n' > "$report_md"
