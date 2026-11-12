# Indian Construction Safety Inspection

Streamlit workspace for visible construction safety screening, regulatory retrieval, and downloadable inspection reports.

The **Model Evaluation** page includes the 457-image PPE benchmark. The detector caught 200 of 213 images with annotated missing PPE and flagged 117 of 244 images without those annotations. See [the evaluation report](outputs/ppe-evaluation/REPORT.md) for the protocol, limitations, and per-image results.

## Run locally

Use Python 3.12 in a virtual environment.

```text
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Indian PPE fine-tune and safety framework

The authors' [PPE compliance project](https://github.com/Sankarlalr23/Detection-of-PPE-Compliance-on-Construction-Site-Using-CV) provides Pascal VOC XML boxes for `NOTSAFE`, `NoHardhat`, `NoJacket`, and `SAFE`. The paper describes images from Indian construction sites and web sources; it reports 2,509 total images, while the public repository contains 2,217 training and 184 validation images. The checked-in text lists disagree with some XML files, so the conversion script uses the paired XML as ground truth and records those discrepancies in `conversion_summary.json`.

The local fine-tuned checkpoint is selected by default in the Streamlit sidebar when available; the original YOLO26n baseline remains selectable. The completed local run and validation report are under `runs/indian-ppe/run-20261110-220848-utc/`. Validation results are not an independent test result. The upstream dataset repository has no license file, so keep the dataset and derived checkpoint local until redistribution rights are confirmed.


Image, batch, video, and camera inspections can use the Pixtral 12B Construction Safety vision model for eight broad hazard groups: fall, electrical, PPE, equipment, structural, cable, public-safety, and environmental hazards. The current YOLO detector remains as a second PPE/object detector. Pixtral findings are matched to the app's existing Indian safety guidance and are shown as unverified visual screening findings; the app does not treat model scores as accident probabilities or legal decisions. Without a configured Pixtral endpoint, inspection runs YOLO only and the UI says broad hazard vision is disconnected.

The Pixtral weights are approximately 25.4 GB. The model card's sample serving setup uses two NVIDIA H200 GPUs, so this model is not bundled into the Streamlit app or expected to run on a typical CPU-only Streamlit host. Run it on a separate GPU server with vLLM and set that server's OpenAI-compatible `/v1` URL as `PIXTRAL_API_BASE`. For example:

```text
vllm serve a1273352/pixtral-12b-construction-safety --tensor-parallel-size 2 --max-model-len 8192 --port 8200
```

For an authenticated Docker server with public HTTPS-ingress and matching Streamlit secrets, see [`deploy/pixtral/README.md`](deploy/pixtral/README.md).

Set `PIXTRAL_API_BASE`, `PIXTRAL_API_KEY` (if the endpoint requires one), and `PIXTRAL_MODEL` in `.streamlit/secrets.toml` locally or the Streamlit deployment secrets. Copy `.streamlit/secrets.toml.example` as a template. Uploaded images and sampled video frames are sent to the configured Pixtral endpoint. Do not put endpoint secrets in Git. The model card reports 92% violation accuracy and 79% violation recall on only 50 hardhat test images; it does not establish category-by-category accuracy for the broader hazard groups. The model card also describes its generated boxes as approximate. Evaluate it on a separate, human-reviewed Indian construction dataset before operational use.

When configured, the optional OpenAI API uses `gpt-4o-mini` by default to write a short text-only comment on returned findings. It receives category names and matched prevention guidance, not uploaded images. It cannot create detections or establish legal compliance. API usage is billed separately from ChatGPT subscriptions.

Image Inspection includes ten generated demo scenes under `demo_samples/`. Use the picker to preview and select one or several scenes. They are synthetic practice inputs, not labeled validation data. With Pixtral connected, these images are submitted to the vision endpoint; demo results are still not accuracy measurements.

Inspection reports include each returned risk type, an uncalibrated model score, immediate corrective action, preventive measures, and relevant source links. YOLO/Pixtral scores are not calibrated as the probability of an accident; they must not be read as an incident probability or legal finding.

After obtaining the authors' dataset, convert and train locally:

```text
python prepare_indian_ppe_yolo.py PATH_TO_REPOSITORY/Data --output data/raw/indian-ppe-yolo-ready
python train_indian_ppe.py --epochs 20 --imgsz 320 --batch 8 --device cpu
```

The app offers the Indian four-class checkpoint as a separate detector after `runs/indian-ppe/*/weights/best.pt` exists. This model predicts person-level compliance categories, not the baseline's 11 PPE object classes. The public data repository has no license file; do not redistribute its images or the derived checkpoint without confirming rights.

## Deploy on Streamlit Community Cloud

Connect the GitHub repository, select branch `main`, set the entrypoint to `app.py`, and choose Python 3.12. BM25 works without external credentials. To enable broad hazard vision, add `PIXTRAL_API_BASE`, `PIXTRAL_API_KEY`, and `PIXTRAL_MODEL` to Streamlit secrets and run Pixtral separately on an accessible GPU endpoint. Add `OPENAI_API_KEY` and optionally `SAFETY_COMMENT_MODEL` only if GPT comments are wanted. The app keeps recent reports in browser-session state. Download a report before closing the session. The Indian fine-tune and raw dataset are excluded from Git because upstream redistribution rights are not documented. Until the fine-tune weights are separately configured, the app uses the bundled baseline PPE/object detector alongside Pixtral when its endpoint is connected.

## Sources

The evaluated YOLO26n model is attributed to [Construction Hazard Detection](https://huggingface.co/yihong1120/Construction-Hazard-Detection) and distributed with its [upstream license](models/yolo/UPSTREAM-LICENSE.txt). The [PPE benchmark](https://data.mendeley.com/datasets/zkzghjvpn2/6) is by Huang and Cheng, CC BY 4.0. The repository contains evaluation records and omits the benchmark photos.

**Opening an evaluated image:** The `image` values in `outputs/ppe-evaluation/predictions.csv` are paths inside the publisher’s dataset ZIP, not GitHub image URLs. The source photos are not stored in this repository. Download the [official version 6 dataset archive](https://data.mendeley.com/public-files/datasets/zkzghjvpn2/files/b8bc8fe2-5aaa-49d8-b012-a9858258f05d/file_downloaded), extract it, and locate the exact image using the CSV’s `image` path. The new `dataset_archive_url` column points to this archive download; it downloads the complete ZIP, not an individual photo.
