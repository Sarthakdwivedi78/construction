# Run Pixtral as a remote inference service

The Streamlit app is already an OpenAI-compatible client for Pixtral. This folder starts the model server on a Linux GPU host; it does not provision a host or create a cloud account.

## Start the server

Use a GPU machine with Docker and the NVIDIA Container Toolkit. The model card's serving example uses two NVIDIA H200 GPUs. It is approximately 25 GB of BF16 weights. Set a strong `PIXTRAL_SERVER_API_KEY` in the GPU provider's secret manager, then run:

```bash
sh deploy/pixtral/start_server.sh
```

The vLLM server listens on port 8200, requires the API key, and exposes the OpenAI-compatible API at `/v1`. Put it behind the GPU provider's HTTPS ingress or a TLS reverse proxy before connecting the public Streamlit app. Do not expose the raw HTTP port to the public internet.

## Connect Streamlit Community Cloud

In the Streamlit app's **Settings → Secrets**, add TOML values:

```toml
PIXTRAL_API_BASE = "https://YOUR-GPU-HOST/v1"
PIXTRAL_API_KEY = "the-same-PIXTRAL_SERVER_API_KEY"
PIXTRAL_MODEL = "a1273352/pixtral-12b-construction-safety"
```

Open the deployed app, select an inspection page, and use **Test Pixtral endpoint**. It should list the configured model before you run image, batch, video, or camera inspections. Uploaded images and sampled frames are sent to that endpoint.

## Cost and availability

GPU pricing depends on provider, region, GPU size, replicas, storage, and whether the endpoint scales to zero. Check the provider's current estimate and set a hard usage limit before enabling a persistent endpoint. A public endpoint is globally reachable, but users will still experience cold starts if workers scale to zero.

The model's own evaluation is limited and does not establish Indian site accuracy across all hazard classes. Keep findings in screening mode and validate on a human-reviewed Indian dataset before operational use.
