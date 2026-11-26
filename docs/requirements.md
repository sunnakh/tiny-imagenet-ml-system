# Tiny ImageNet classifier requirements

**User and decision.** An image-labeling reviewer needs a proposed class for a 64 × 64 image. The service returns one of the Tiny ImageNet classes or abstains so a reviewer can handle the image. Inputs are JPEG or PNG bytes up to 64 KiB, including grayscale images. Responses carry the prediction ID and release ID for later feedback.

**ML task and baseline.** Supervised 200-class classification from scratch. The non-ML baseline always predicts the most frequent class in the fit partition, breaking ties by class index. The model must beat that baseline in accuracy and macro F1. On the same version of the release holdout, it must match or beat the champion in both metrics and improve at least one.

**Data contract.** Pin the Tiny ImageNet source revision. Verify class balance and IDs, decode images and convert to RGB, quarantine invalid records, then write deterministic fit, tuning, and gate Parquet partitions. Compute normalization exclusively on fit images. Use tuning accuracy for checkpoint selection and the gate holdout for release decisions. The original Tiny ImageNet validation split is reserved for the research notebook's final evaluation.

**Serving and operation.** Target HTTP p95 at or below 250 ms for 50 sequential warmed requests on the local machine. Return a structured fallback with HTTP 503 if the model is missing, busy, or exceeds the inference timeout. Track response latency, fallback rate, RGB mean shift, and accuracy and macro F1 on supplied feedback. Trace predictions through the release ID, MLflow run and registry version, code fingerprint, and data version. Record the Git commit when the source tree has one.

**Acceptance checks.**

- Run the fixture end to end from data preparation to an HTTP response and monitoring.
- Produce byte-identical data artifacts given the same input, seed, and environment. Quarantine an injected malformed record.
- Reject a deliberately worse candidate and retain the active release.
- Meet the warmed HTTP p95 target and reproduce the service fallback by removing the active weights from a disposable release.
- Trigger an alert with shifted inputs and demonstrate shadow traffic, promotion, and rollback.
- Validate the deployment definition, deploy an immutable container revision to the authorized cloud target, verify it with an authenticated request, and tear it down.

**Design check.** Explain how the release holdout can become less informative after repeated comparisons and how to rotate it without comparing scores computed on different datasets. Then compare the answer against the assigned reading and record any gaps.
