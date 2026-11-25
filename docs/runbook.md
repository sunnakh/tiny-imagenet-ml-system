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

Feedback is optional. A trusted reviewer sends a request ID and verified integer class to the feedback endpoint. The API accepts an identical retry and rejects conflicting labels.

```bash
curl -fsS http://127.0.0.1:8080/feedback \
  -H 'Content-Type: application/json' \
  --data '{"request_id":"REPLACE_WITH_REQUEST_ID","label":0}'
```
