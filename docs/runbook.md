# Classifier runbook

Open `/dashboard` or request `/metrics`. Metrics use the last hour, capped at 1,000 records per release. The drift and labeled-quality alerts require at least 20 observations. Wait for relevant samples after switching releases.

| Alert | Investigate | Recovery |
| --- | --- | --- |
| `fallback_rate_high` | Check `/health`, the current release, model file availability and service logs. | Restore a missing bundle from a verified export or run rollback if a previous release exists. Confirm that predictions report `mode: active`. |
| `latency_high` | Run the warmed HTTP load test. Inspect model-load time, request rate and CPU saturation. | If latency rose after promotion, roll back and rerun the same workload. |
| `input_drift` | Compare image dimensions, capture conditions and RGB mean shift against the fit reference. | Repair an upstream preprocessing change and verify on representative requests. Collect fresh labels if the shift is real. |
| `feedback_accuracy_low` | Check label provenance and sample size. Confirm whether the drop began with a promotion. | Roll back where appropriate. Check incoming feedback on the restored model. |

Use the same state directory as the server for release changes:

```bash
.venv/bin/python -m tiny_imagenet --state .state/full rollback
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/metrics
```

The automated incident test uses a disposable export. It sends shifted images, takes the weights offline, observes the fallback alert, and restores the file. A second test sends shadow traffic and switches back to the previous release. Simulate incidents in a fixture or a disposable copy.

## Local monitoring stack

Use the existing smoke model to exercise the HTTP service and monitoring. In one terminal, export a fresh copy and start the service:

```bash
.venv/bin/python -m tiny_imagenet --state .state/smoke export .state/monitoring-smoke
.venv/bin/python -m tiny_imagenet --state .state/monitoring-smoke serve --port 18080
```

In a second terminal, start the stack on Docker Desktop:

```bash
docker compose -f infra/monitoring/compose.yml up -d
```

Open the [Grafana dashboard](http://127.0.0.1:13000/d/tiny-imagenet-service), check the [Prometheus target](http://127.0.0.1:19090/targets), and view [Alertmanager](http://127.0.0.1:19093). Prometheus scrapes `/metrics/prometheus` every five seconds. The existing `/metrics` endpoint still returns JSON. The exported metrics describe the active release's trailing one-hour window of at most 1,000 predictions. The duration gauge represents server-side processing p95. Accuracy and macro F1 appear after 20 labels.

Run both client-side latency workloads:

```bash
.venv/bin/python -m tiny_imagenet.load_test --url http://127.0.0.1:18080 --requests 50
.venv/bin/python -m tiny_imagenet.load_test --url http://127.0.0.1:18080 --requests 100 --concurrency 4
```

The 250 ms target applies to the sequential workload with zero fallbacks. For the concurrent workload, use both active-response p95 and the fallback rate. The service has one inference slot. A request that arrives while the slot is busy can immediately abstain with HTTP 503.

For a local incident drill, temporarily rename the active `weights.pt` inside the disposable export, send a prediction, and verify HTTP 503. Restore the file and verify HTTP 200. Follow the notifications with:

```bash
docker compose -f infra/monitoring/compose.yml logs -f webhook
```

The local receiver logs a firing and a resolved `TinyImageNetModelUnavailable` notification. The window fallback alert can remain active until the failed attempts age out or the fallback ratio falls to 10% or less. To stop the stack, run `docker compose -f infra/monitoring/compose.yml down` and stop the local API process.

## Feedback

Feedback is optional. A trusted reviewer sends a request ID and verified integer class to the feedback endpoint. The API accepts an identical retry and rejects conflicting labels.

```bash
curl -fsS http://127.0.0.1:8080/feedback \
  -H 'Content-Type: application/json' \
  --data '{"request_id":"REPLACE_WITH_REQUEST_ID","label":0}'
```
