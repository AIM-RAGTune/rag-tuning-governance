# Kubernetes

These examples run RAGTune as a finite Kubernetes Job or CronJob. Replace placeholder image and storage values before use.

```bash
kubectl apply -k deploy/kubernetes
```

The examples do not include raw datasets or credentials. Mount approved data separately when a job requires it.
