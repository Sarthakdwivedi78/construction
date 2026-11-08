# Indian Construction Safety Inspection

Streamlit workspace for visible construction safety screening, regulatory retrieval, and downloadable inspection reports.

The **Model Evaluation** page includes the 457-image PPE benchmark. The detector caught 200 of 213 images with annotated missing PPE and flagged 117 of 244 images without those annotations. See [the evaluation report](outputs/ppe-evaluation/REPORT.md) for the protocol, limitations, and per-image results.

## Run locally

Use Python 3.12 in a virtual environment.

```text
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Connect the GitHub repository, select branch `main`, set the entrypoint to `app.py`, and choose Python 3.12. The release includes the evaluated CPU checkpoint. BM25 works without external credentials. Optional vision requires `OPENAI_API_KEY` and `SAFETY_VISION_MODEL` in the deployment's secrets. The app keeps recent reports in browser-session state. Download a report before closing the session.

## Sources

The evaluated YOLO26n model is attributed to [Construction Hazard Detection](https://huggingface.co/yihong1120/Construction-Hazard-Detection) and distributed with its [upstream license](models/yolo/UPSTREAM-LICENSE.txt). The [PPE benchmark](https://data.mendeley.com/datasets/zkzghjvpn2/6) is by Huang and Cheng, CC BY 4.0. The repository contains evaluation records and omits the benchmark photos.
