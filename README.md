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

The app’s knowledge corpus contains the OSHWC Code, 2020, which commenced on 21 November 2025, and the OSHWC (Central) Rules, 2026, published on 8 May 2026. The 2026 Central Rules supersede the BOCW Central Rules, 1998 in the corpus; historical references remain marked as historical. State-specific requirements and the appropriate government must still be checked for each site. See the [Ministry’s Labour Codes page](https://www.labour.gov.in/offerings/schemes-and-services/details/labour-codes-gzNzQzMtQWa), the [PIB commencement announcement](https://www.pib.gov.in/newsite/erelcontent.aspx?lang=2&reg=48&relid=279088), and the [Ministry’s 2026 Rules PDF](https://www.labour.gov.in/static/uploads/2026/05/ee246f790cad0b8e99c3828f34fa09a6.pdf). App output is visual screening support and does not certify compliance or site safety.

Inspection reports include each detected risk type, an uncalibrated visual-cue score, immediate corrective action, preventive measures, and relevant source links. YOLO confidence is not calibrated as the probability of an accident; it must not be read as an incident probability or legal finding.

After obtaining the authors' dataset, convert and train locally:

```text
python prepare_indian_ppe_yolo.py PATH_TO_REPOSITORY/Data --output data/raw/indian-ppe-yolo-ready
python train_indian_ppe.py --epochs 20 --imgsz 320 --batch 8 --device cpu
```

The app offers the Indian four-class checkpoint as a separate detector after `runs/indian-ppe/*/weights/best.pt` exists. This model predicts person-level compliance categories, not the baseline's 11 PPE object classes. The public data repository has no license file; do not redistribute its images or the derived checkpoint without confirming rights.

## Deploy on Streamlit Community Cloud

Connect the GitHub repository, select branch `main`, set the entrypoint to `app.py`, and choose Python 3.12. BM25 works without external credentials. Optional vision requires `OPENAI_API_KEY` and `SAFETY_VISION_MODEL` in the deployment's secrets. The app keeps recent reports in browser-session state. Download a report before closing the session. The Indian fine-tune and raw dataset are excluded from Git because upstream redistribution rights are not documented. To enable the fine-tune on a server, first obtain permission to deploy the checkpoint, then provide it through the `INDIAN_PPE_WEIGHTS` environment variable or an approved private artifact store. Until that permission is confirmed, the public app falls back to the separately licensed baseline model.

## Sources

The evaluated YOLO26n model is attributed to [Construction Hazard Detection](https://huggingface.co/yihong1120/Construction-Hazard-Detection) and distributed with its [upstream license](models/yolo/UPSTREAM-LICENSE.txt). The [PPE benchmark](https://data.mendeley.com/datasets/zkzghjvpn2/6) is by Huang and Cheng, CC BY 4.0. The repository contains evaluation records and omits the benchmark photos.

**Opening an evaluated image:** The `image` values in `outputs/ppe-evaluation/predictions.csv` are paths inside the publisher’s dataset ZIP, not GitHub image URLs. The source photos are not stored in this repository. Download the [official version 6 dataset archive](https://data.mendeley.com/public-files/datasets/zkzghjvpn2/files/b8bc8fe2-5aaa-49d8-b012-a9858258f05d/file_downloaded), extract it, and locate the exact image using the CSV’s `image` path. The new `dataset_archive_url` column points to this archive download; it downloads the complete ZIP, not an individual photo.
