import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

import cv2
import streamlit as st
from detection import detect, frames, load_detector
from measurements import parse_measurements
from reasoning import Vision, inspect, report
from reporting import pdf
from retrieval import Index, ROOT

@st.cache_resource
def detector():
    path = ROOT / "models" / "yolo" / "yolo26n.pt"
    with path.open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    if digest != "80c463daf7cf151f20e92231276162607bc4611c013cfb6e052b89442b764b1d":
        raise ValueError("The bundled model checksum did not match the evaluated checkpoint.")
    model = load_detector(path)
    model.to("cpu")
    return model

@st.cache_resource
def index(hybrid):
    return Index(hybrid)

st.set_page_config(page_title="Construction Safety Inspection", layout="wide")
st.title("Indian Construction Site Safety Inspection")
st.caption("Screening of visible conditions and retrieval of Indian safety provisions.")
page = st.sidebar.radio("Workspace", ["Home", "Live Inspection", "Image Inspection", "Video Inspection",
                       "Regulatory Knowledge Base", "Retrieval Testing", "Model Evaluation",
                       "Training Dashboard", "Inspection History"])
if page == "Home":
    st.write("Upload an image or sample a video to screen visible construction hazards.")
    st.write("Start with Image Inspection or explore the regulatory corpus in the sidebar.")
    st.stop()
if page == "Training Dashboard":
    runs = sorted((ROOT / "models" / "checkpoints").glob("*/results.csv"))
    if not runs:
        st.info("Training runs appear here after validation and optional YOLO fine-tuning.")
    for run in runs:
        with st.expander(run.parent.name):
            import pandas as pd
            st.dataframe(pd.read_csv(run))
    st.stop()
if page == "Regulatory Knowledge Base":
    st.dataframe(index(False).rows, width="stretch")
    st.stop()
if page == "Retrieval Testing":
    taxonomy = json.loads((ROOT / "taxonomy.json").read_text(encoding="utf-8-sig"))
    category = st.selectbox("Hazard category", list(taxonomy))
    authority = st.selectbox("Appropriate government", ["Unknown", "Central", "State"])
    state = st.text_input("State or union territory", "West Bengal")
    query = st.text_input("Regulatory query", taxonomy[category]["query"])
    dense_available = all(importlib.util.find_spec(name) for name in ("faiss", "sentence_transformers"))
    hybrid = st.checkbox("Enable BGE and FAISS", disabled=not dense_available,
                         help="Dense retrieval requires the optional hybrid dependencies.")
    if st.button("Search"):
        st.json(index(hybrid).search(query, category, authority, state=state))
    st.stop()
if page == "Model Evaluation":
    from evaluate import retrieval

    benchmark_dir = Path(os.environ.get("SAFETY_PPE_EVAL_DIR") or ROOT / "outputs" / "ppe-evaluation")
    summary_path = benchmark_dir / "summary.json"
    if summary_path.is_file():
        result = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        metrics = result["metrics"]
        st.subheader("Published PPE image benchmark")
        st.caption(
            f'{result["images"]} annotated images from the publisher’s validation split. '
            f'Checkpoint: {result["checkpoint"]} · Confidence threshold: {result["threshold"]:.2f}'
        )
        st.write(
            f'{result["positive_images"]} images with an annotated missing helmet or vest. '
            f'{result["negative_images"]} images without either missing-PPE annotation.'
        )
        recall, precision, specificity, f1 = st.columns(4)
        for column, label, value in (
            (recall, "Hazard recall", metrics["recall"]),
            (precision, "Alert precision", metrics["precision"]),
            (specificity, "Specificity", metrics["specificity"]),
            (f1, "F1", metrics["f1"]),
        ):
            column.metric(label, f"{value:.1%}" if value is not None else "—")
        st.dataframe(
            [
                {"Published annotation": "Missing PPE", "Alert": metrics["tp"], "No alert": metrics["fn"]},
                {"Published annotation": "No annotated missing PPE", "Alert": metrics["fp"], "No alert": metrics["tn"]},
            ],
            width="stretch",
            hide_index=True,
        )
        st.warning(
            f'{metrics["fn"]} annotated missing-PPE images produced no alert. '
            'A no-alert result leaves site safety undetermined.'
        )
        st.caption(result["scope"])
        st.markdown("[Publisher dataset and license](https://data.mendeley.com/datasets/zkzghjvpn2/6)")
        with st.expander("Per-cue metrics, timing and evidence gate"):
            st.json({k: result[k] for k in ("by_cue", "box_localization", "latency_ms", "evidence_gate")}, expanded=False)
        st.download_button("Download benchmark summary", summary_path.read_bytes(), "ppe-summary.json", "application/json")
        per_image = benchmark_dir / "predictions.csv"
        if per_image.is_file():
            st.download_button("Download per-image results", per_image.read_bytes(), "ppe-predictions.csv", "text/csv")
        st.divider()

    st.caption("Authored retrieval checks for the seed corpus")
    evaluation = ROOT / "outputs" / "evaluation"
    completed = [json.loads(path.read_text(encoding="utf-8-sig"))
                 for path in sorted(evaluation.glob("*.json"))]
    if completed:
        st.dataframe(completed, width="stretch")
        st.bar_chart(completed, x="mode", y="latency_ms")
    if st.button("Run BM25 checks"):
        st.json(retrieval(False, 3))
    st.stop()
if page == "Inspection History":
    st.caption("History is available for this browser session. Download a report to retain it.")
    history = st.session_state.get("inspection_history", [])
    if not history:
        st.info("No saved inspections yet. Complete an image or video inspection to add one.")
    query = st.text_input("Filter saved inspections", "").strip().casefold()
    for entry in history:
        haystack = " ".join((entry.get("inspection_id", ""), entry.get("site_type", ""),
                             entry.get("overall_risk_level", ""), entry.get("created_at", ""))).casefold()
        if query and query not in haystack:
            continue
        title = f'{entry.get("created_at", "Unknown time")} | {entry.get("site_type", "Unspecified")} | {entry.get("overall_risk_level", "Undetermined")}'
        with st.expander(title):
            st.json(entry)
            stem = entry.get("inspection_id", "inspection")
            st.download_button("Download JSON", json.dumps(entry, indent=2),
                               f"{stem}.json", key=f"history-json-{stem}")
            st.download_button("Download PDF", pdf(entry), f"{stem}.pdf",
                               "application/pdf", key=f"history-pdf-{stem}")
    st.stop()
if page == "Live Inspection":
    media = st.camera_input("Capture a site image")
else:
    formats = ["mp4", "mov", "avi"] if page == "Video Inspection" else ["jpg", "jpeg", "png"]
    media = st.file_uploader("Site media", type=formats)
with st.sidebar:
    site = st.text_input("Site type", "Building construction")
    activity = st.text_input("Work activity")
    state = st.text_input("State or union territory", "West Bengal")
    authority = st.selectbox("Appropriate government", ["Unknown", "Central", "State"])
    date = st.date_input("Inspection date")
    measurements = st.file_uploader("Optional measured site evidence", type=["json"])
    checks = st.file_uploader("Optional inspector review records", type=["json"])
    st.caption("YOLO26n · Bundled benchmark checkpoint")
    dense_available = all(importlib.util.find_spec(name) for name in ("faiss", "sentence_transformers"))
    hybrid = st.checkbox("Enable BGE and FAISS", disabled=not dense_available,
                         help="Dense retrieval requires the optional hybrid dependencies.")
    vision_configured = bool(os.getenv("OPENAI_API_KEY") and os.getenv("SAFETY_VISION_MODEL"))
    provider = st.selectbox("Vision provider", ["Off", "OpenAI API"] if vision_configured else ["Off"])
    if provider == "OpenAI API":
        st.caption("Sends sampled images and site context to the configured API.")
    interval = st.number_input("Video sampling interval in seconds", 1, 60, 2)
    limit = st.number_input("Maximum video frames", 1, 60, 12)
if st.button("Run inspection", disabled=media is None):
    st.session_state.pop("result", None)
    try:
        model = detector()
        kb = index(hybrid)
        if provider == "OpenAI API" and not os.getenv("SAFETY_VISION_MODEL"):
            raise ValueError("Set SAFETY_VISION_MODEL to a vision-capable Responses API model.")
        vision = Vision(os.environ["SAFETY_VISION_MODEL"]) if provider == "OpenAI API" else None
        taxonomy = json.loads((ROOT / "taxonomy.json").read_text(encoding="utf-8-sig"))
        meta = {"site_type": site, "activity": activity, "state": state,
                "authority": authority, "inspection_date": date.isoformat()}
        if measurements:
            meta["site_measurements"] = parse_measurements(measurements.getvalue())
        if checks:
            review = json.loads(checks.getvalue())
            if not isinstance(review, dict) or any(not isinstance(row, dict) for row in review.values()):
                raise ValueError("Inspector review records must map provision IDs to record objects.")
            meta["reviewed_checks"] = review
        start = perf_counter()
        records, previews = [], []
        suffix = Path(media.name).suffix.lower()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / ("upload" + suffix)
            target.write_bytes(media.getbuffer())
            with st.spinner("Inspecting selected media"):
                for timestamp, frame in frames(target, suffix in {".mp4", ".mov", ".avi"}, interval, limit):
                    objects, annotated = detect(model, frame)
                    ok, original = cv2.imencode(".jpg", frame)
                    if not ok:
                        raise ValueError("Could not encode frame.")
                    hazards, warnings = inspect(objects, kb, meta, taxonomy, original.tobytes(), vision)
                    records.append(dict(timestamp_seconds=timestamp, objects=objects,
                                        hazards=hazards, warnings=warnings))
                    ok, preview = cv2.imencode(".jpg", annotated)
                    if ok:
                        previews.append((timestamp, preview.tobytes()))
        output = report(meta, records, perf_counter() - start)
        output["retrieval_mode"] = "BM25 + BGE" if hybrid else "BM25"
        output["vision_enabled"] = vision is not None
        output["vision_provider"] = provider
        output["vision_model"] = vision.model if vision else None
        from datetime import datetime, timezone
        output["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        history = st.session_state.get("inspection_history", [])
        st.session_state["inspection_history"] = [output, *history][:20]
        st.session_state["result"] = output, previews
    except Exception as exc:
        st.error(f"Inspection could not run ({type(exc).__name__}). Check model, dependencies and media.")
        if isinstance(exc, (FileNotFoundError, ValueError)):
            st.caption(str(exc))
if "result" in st.session_state:
    output, previews = st.session_state["result"]
    st.subheader(output["overall_risk_level"])
    for timestamp, preview in previews:
        with st.expander(f"Detections at {timestamp:.2f} s"):
            st.image(preview)
    for item in output["identified_hazards"]:
        with st.expander(item["hazard"] + " | " + item["violation_status"]):
            st.write(item["visual_evidence"], item["reasoning"], item["corrective_action"])
            for source in item["retrieved_sources"]:
                st.write(source["indian_regulation"], source["provision_number"],
                         source["verification_status"], source["source_url"])
            if item["model_assessment"]:
                st.caption("Unreviewed vision model assessment")
                st.json(item["model_assessment"])
    st.json(output, expanded=False)
    st.download_button("Download JSON", json.dumps(output, indent=2), "inspection.json")
    st.download_button("Download PDF", pdf(output), "inspection.pdf", "application/pdf")
