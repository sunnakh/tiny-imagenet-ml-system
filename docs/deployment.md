# Container deployment

Export an active release into a fresh `release/` folder. Use `.state/fixture` here only for a synthetic demonstration. Build an image for the Cloud Run Linux target and push it to an existing Artifact Registry repository. Use the immutable image digest when applying Terraform. [Cloud Run container contract](https://docs.cloud.google.com/run/docs/container-contract)

```bash
.venv/bin/python -m tiny_imagenet --state .state/full export release
docker buildx build --platform linux/amd64 --push -t "$IMAGE_TAG" .
```

The Terraform configuration expects an existing Google Cloud project with Cloud Run enabled, a runtime service account, deployment credentials, and caller IAM access. It creates one authenticated service. Set `TF_VAR_project_id`, `TF_VAR_image` to the pushed digest reference, and `TF_VAR_runtime_service_account` to the service account’s email. An optional `TF_VAR_region` overrides the default region.

```bash
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform validate
terraform -chdir=infra/terraform plan
terraform -chdir=infra/terraform apply
SERVICE_URL="$(terraform -chdir=infra/terraform output -raw service_url)"
curl -fsS -H "Authorization: Bearer $(gcloud auth print-identity-token)" "$SERVICE_URL/health"
curl -fsS -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H 'Content-Type: application/octet-stream' --data-binary @image.png "$SERVICE_URL/predict"
terraform -chdir=infra/terraform destroy
```

After teardown, check that the service is absent. Image storage belongs to the existing registry. Remove the demo image through your registry process when it is no longer needed. Cloud rollbacks use the previously verified image digest. The SQLite database in the demo instance is ephemeral. Persistent feedback and monitoring require an external data store.
