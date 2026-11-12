import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

import cv2
import streamlit as st
from detection import INDIAN_PPE_LABELS, detect, frames, load_detector
from measurements import parse_measurements
from reasoning import OpenAICommentary, PixtralHazardVision, explain_finding, inspect, report
from reporting import pdf
from retrieval import Index, ROOT

def indian_ppe_checkpoint():
    configured = os.getenv("INDIAN_PPE_WEIGHTS", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    candidates = sorted((ROOT / "runs" / "indian-ppe").glob("*/weights/best.pt"),
                        key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None

def assessment_detail(item):
    return explain_finding(item)

def assessment_label(item):
    if item.get("assessment_display_status"):
        return item["assessment_display_status"]
    suggestion = item.get("model_assessment")
    if isinstance(suggestion, dict) and suggestion.get("status"):
        return f'AI suggestion: {suggestion["status"]} · needs human review'
    if item.get("violation_status") == "Insufficient Evidence":
        return "Needs site verification"
    return item.get("violation_status", "Needs site verification")

class DemoMedia:
    def __init__(self, path, category):
        self.name = f"DEMO_{category}.png"
        self._content = path.read_bytes()

    def getbuffer(self):
        return memoryview(self._content)

def annotate_hazards(image, hazards):
    for item in hazards:
        box = item.get("evidence_box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in box]
        color = (30, 110, 235) if item.get("observation_origin") == "pixtral_vision" else (20, 180, 230)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = item.get("model_category") or item.get("hazard", "risk")
        score = item.get("confidence")
        if isinstance(score, (int, float)):
            label = f"{label} {float(score):.2f}*"
        cv2.putText(image, label[:80], (max(0, x1), max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
    return image

def setting(name, default=""):
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or os.getenv(name, default)

def category_label(item):
    category = item.get("category") or item.get("risk_type") or "site safety"
    labels = {"ppe": "Personal protective equipment (PPE)", "height": "Work at height",
              "scaffold": "Scaffolding", "confined": "Confined space",
              "operations": "Site operations"}
    return labels.get(category, str(category).replace("_", " ").title())

@st.cache_resource
def detector(model_choice):
    if model_choice == "Indian PPE fine-tune (4 classes)":
        path = indian_ppe_checkpoint()
        if path is None:
            raise FileNotFoundError("Set INDIAN_PPE_WEIGHTS to a valid checkpoint or keep a completed run under runs/indian-ppe/.")
        model = load_detector(path, INDIAN_PPE_LABELS)
    else:
        path = ROOT / "models" / "yolo" / "yolo26n.pt"
        digest = hashlib.sha256()
        with path.open("rb") as weights:
            for block in iter(lambda: weights.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != "80c463daf7cf151f20e92231276162607bc4611c013cfb6e052b89442b764b1d":
            raise ValueError("The bundled model checksum did not match the evaluated checkpoint.")
        model = load_detector(path)
    model.to("cpu")
    return model

@st.cache_resource
def index(hybrid):
    return Index(hybrid)

st.set_page_config(page_title="Construction Safety Inspection", layout="wide")
st.title("Indian Construction Site Safety Inspection")
st.caption("Broad construction-hazard screening with Pixtral 12B when connected, YOLO PPE/object cues, and retrieval from the Indian safety corpus. Every finding needs human review.")
page = st.sidebar.radio("Workspace", ["Home", "Live Inspection", "Image Inspection", "Video Inspection",
                       "Regulatory Knowledge Base", "Retrieval Testing", "Model Evaluation",
                       "Training Dashboard", "Inspection History"])
indian_weights = indian_ppe_checkpoint()
model_options = ["Baseline YOLO26n"]
if indian_weights:
    model_options.append("Indian PPE fine-tune (4 classes)")
model_choice = st.sidebar.selectbox("Detection model", model_options,
                                    index=1 if indian_weights else 0)
if page == "Home":
    st.subheader("Current Indian safety framework")
    st.write("The Occupational Safety, Health and Working Conditions Code, 2020 came into force on 21 November 2025. The Occupational Safety, Health and Working Conditions (Central) Rules, 2026 were published on 8 May 2026 and supersede the BOCW Central Rules, 1998. State-specific notifications and site applicability still need review.")
    st.markdown("[Ministry of Labour: Labour Codes, current Central Rules and notices](https://www.labour.gov.in/offerings/schemes-and-services/details/labour-codes-gzNzQzMtQWa) · [PIB: Labour Codes effective from 21 November 2025](https://www.pib.gov.in/newsite/erelcontent.aspx?lang=2&reg=48&relid=279088)")
    st.warning("This app screens visual cues; it cannot determine legal compliance or certify that a site is safe. Confirm the site, activity, appropriate government and applicable State requirements with a qualified safety professional.")
    st.subheader("Indian PPE model")
    if indian_weights:
        st.success(f"Indian PPE fine-tune selected by default · classes: {', '.join(INDIAN_PPE_LABELS)}")
        india_eval = indian_weights.parent.parent / "best_validation.json"
        if india_eval.is_file():
            evaluation = json.loads(india_eval.read_text(encoding="utf-8"))
            scores = evaluation["overall"]
            cols = st.columns(4)
            for col, label, key in zip(cols, ("Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"),
                                       ("precision", "recall", "map50", "map50_95")):
                col.metric(label, f"{scores[key]:.3f}")
            st.caption("Validation only: 184 images and 332 boxes, including 18 NoHardhat boxes. This is not an independent test set; the data includes Indian-site and web imagery.")
        st.caption(f"Checkpoint: {indian_weights}")
    else:
        st.info("No local Indian PPE checkpoint is configured. The app is using the original baseline. Set INDIAN_PPE_WEIGHTS to the trained checkpoint to enable the Indian fine-tune.")
    st.caption("The PIB announcement confirms the Code’s effective date. The Ministry page lists the Code and 2026 Central Rules. These official HTML pages avoid relying on the direct PDF links that were failing.")
    st.write("Choose Image Inspection, Video Inspection or Live Inspection to screen site media. Use the sidebar to select the site state and work context before retrieval.")
    st.stop()
if page == "Training Dashboard":
    if indian_weights:
        run_dir = indian_weights.parent.parent
        manifest_path = run_dir / "training_manifest.json"
        metrics_path = run_dir / "metrics.json"
        if manifest_path.is_file() and metrics_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))
            scores = summary["validation_of_selected_best_checkpoint"]["overall"]
            st.subheader("Indian PPE fine-tune")
            st.caption(f"{manifest['completed_epochs']} epochs · {manifest['dataset_conversion_counts']['train']['images']:,} training images · {manifest['dataset_conversion_counts']['val']['images']:,} validation images · classes: {', '.join(manifest['trained_classes'])}")
            cols = st.columns(4)
            for col, label, key in zip(cols, ("Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"),
                                       ("precision", "recall", "map50", "map50_95")):
                col.metric(label, f"{scores[key]:.3f}")
            st.warning("These are validation results, not an independent test. The validation set has only 18 NoHardhat boxes.")
            curve = run_dir / "results.png"
            if curve.is_file():
                st.image(str(curve), caption="Training and validation curves")
            report_path = run_dir / "report.md"
            if report_path.is_file():
                st.download_button("Download Indian PPE training report", report_path.read_bytes(),
                                   "indian-ppe-training-report.md", "text/markdown")
    runs = sorted((ROOT / "models" / "checkpoints").glob("*/results.csv"))
    if not runs:
        st.info("Training runs appear here after validation and optional YOLO fine-tuning.")
    for run in runs:
        with st.expander(run.parent.name):
            import pandas as pd
            st.dataframe(pd.read_csv(run))
    st.stop()
if page == "Regulatory Knowledge Base":
    st.dataframe(index(False).rows, use_container_width=True)
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
    if indian_weights:
        run_dir = indian_weights.parent.parent
        validation_path = run_dir / "best_validation.json"
        if validation_path.is_file():
            result = json.loads(validation_path.read_text(encoding="utf-8"))
            st.subheader("Indian PPE fine-tune validation")
            st.caption("Author repository's held-out validation split · 184 images · 332 boxes · checkpoint selected from this same split")
            scores = result["overall"]
            cols = st.columns(4)
            for col, label, key in zip(cols, ("Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"),
                                       ("precision", "recall", "map50", "map50_95")):
                col.metric(label, f"{scores[key]:.3f}")
            metrics_doc = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            class_counts = metrics_doc["dataset"]["validation"]["classes"]
            st.dataframe([
                {"Class": name, "Validation boxes": class_counts.get(name, 0),
                 "Precision": values["precision"], "Recall": values["recall"],
                 "mAP@0.5": values["map50"], "mAP@0.5:0.95": values["map50_95"]}
                for name, values in result["classes"].items()
            ], hide_index=True, use_container_width=True)
            st.warning("Validation results are not a separate test result. Scores may be optimistic because this split was also used to select the best checkpoint.")
            st.download_button("Download Indian PPE metrics", (run_dir / "metrics.json").read_bytes(),
                               "indian-ppe-metrics.json", "application/json")
            st.divider()
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
            use_container_width=True,
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
        st.dataframe(completed, use_container_width=True)
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
    media_items = [media] if media is not None else []
else:
    formats = ["mp4", "mov", "avi"] if page == "Video Inspection" else ["jpg", "jpeg", "png"]
    multiple_images = page == "Image Inspection"
    media = st.file_uploader("Site media", type=formats, accept_multiple_files=multiple_images)
    media_items = media if multiple_images else ([media] if media is not None else [])
    if multiple_images and len(media_items) > 1:
        st.caption(f"{len(media_items)} images selected. Each image will receive its own inspection and downloadable report.")
with st.sidebar:
    site = st.text_input("Site type", "Building construction")
    activity = st.text_input("Work activity")
    states_and_uts = ["West Bengal", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
                      "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
                      "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
                      "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
                      "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "Andaman and Nicobar Islands",
                      "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir",
                      "Ladakh", "Lakshadweep", "Puducherry", "Unknown / not selected"]
    state = st.selectbox("State or union territory", states_and_uts)
    authority = st.selectbox("Appropriate government", ["Unknown", "Central", "State"])
    date = st.date_input("Inspection date")
    measurements = st.file_uploader("Optional measured site evidence", type=["json"])
    checks = st.file_uploader("Optional inspector review records", type=["json"])
    st.caption(f"Selected detection model: {model_choice}")
    st.caption("Central and State applicability can differ; confirm the provision for the actual site and work activity.")
    dense_available = all(importlib.util.find_spec(name) for name in ("faiss", "sentence_transformers"))
    hybrid = st.checkbox("Enable BGE and FAISS", disabled=not dense_available,
                         help="Dense retrieval requires the optional hybrid dependencies.")
    api_key = setting("OPENAI_API_KEY")
    comment_model_name = setting("SAFETY_COMMENT_MODEL", "gpt-4o-mini")
    comment_options = ["Off", "OpenAI API"] if api_key else ["Off"]
    provider = st.selectbox("GPT comments on model findings", comment_options,
                            index=1 if api_key else 0,
                            key="ai_commentary_provider",
                            help="OpenAI writes a short text advisory for findings already returned by a vision model. It does not add detections.")
    pixtral_base_url = setting("PIXTRAL_API_BASE").strip()
    pixtral_api_key = setting("PIXTRAL_API_KEY", "not-needed")
    pixtral_model_name = setting("PIXTRAL_MODEL", "a1273352/pixtral-12b-construction-safety")
    if pixtral_base_url:
        st.info(f"Pixtral endpoint configured · {pixtral_model_name}")
        st.caption("The endpoint is checked only when you run an inspection or press the test button. Uploaded images/video frames are sent to this endpoint. YOLO remains an additional PPE/object detector.")
        if st.button("Test Pixtral endpoint", key="test_pixtral_endpoint"):
            try:
                available_models = PixtralHazardVision(
                    pixtral_base_url, pixtral_api_key, pixtral_model_name
                ).test_endpoint()
                if pixtral_model_name in available_models:
                    st.success(f"Endpoint reachable; configured model is served: {pixtral_model_name}")
                else:
                    st.warning("Endpoint reachable, but the configured model ID was not listed. Served IDs: "
                               + (", ".join(available_models) or "none"))
            except Exception as exc:
                st.error(f"Pixtral endpoint test failed ({type(exc).__name__}): "
                         + (" ".join(str(exc).split())[:240] or "No additional details."))
    else:
        st.warning("Pixtral is not configured. Inspections will use YOLO-only mode until PIXTRAL_API_BASE points to a reachable Pixtral vLLM endpoint.")
        st.caption("For a Streamlit Cloud deployment, the Pixtral server must be publicly reachable from the Streamlit app. localhost on your computer will not work from Streamlit Cloud.")
    st.caption("OpenAI comments receive detected category names and prevention text only; uploaded images are not sent to OpenAI. The Indian YOLO fine-tune detects four PPE classes: NOTSAFE, NoHardhat, NoJacket, SAFE.")
    if provider == "OpenAI API":
        st.caption(f"Comment model: {comment_model_name}. Comments are advisory; the knowledge base supplies prevention measures and a site professional must verify findings.")
    else:
        st.info("To enable GPT comments, set OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable. API usage is billed separately from ChatGPT.")
    interval = st.number_input("Video sampling interval in seconds", 1, 60, 2)
    limit = st.number_input("Maximum video frames", 1, 60, 12)
if page == "Image Inspection":
    demo_manifest_path = ROOT / "demo_samples" / "manifest.json"
    demo_manifest = json.loads(demo_manifest_path.read_text(encoding="utf-8")) if demo_manifest_path.is_file() else []
    if demo_manifest:
        with st.expander(f"Try {len(demo_manifest)} built-in hazard examples", expanded=True):
            demo_by_name = {row["category"]: row for row in demo_manifest}
            selected_demo_names = st.multiselect(
                "Select demo scenes to run",
                list(demo_by_name),
                help="Choose one scene or a small batch. Each selected scene gets its own result and report.")
            preview_index = list(demo_by_name).index(selected_demo_names[0]) if selected_demo_names else 0
            preview_name = st.selectbox("Preview a sample scene", list(demo_by_name), index=preview_index)
            preview_row = demo_by_name[preview_name]
            preview_path = ROOT / "demo_samples" / preview_row["filename"]
            st.image(preview_path.read_bytes(), caption=f'{preview_name} · synthetic demo scene')
            st.caption("These generated examples are for app practice, not labeled test data or accuracy evaluation.")
            if selected_demo_names:
                if pixtral_base_url:
                    st.caption("Pixtral analyzes the selected image for its eight broad hazard categories. Synthetic examples are practice inputs, not accuracy test data.")
                else:
                    st.caption("Without a configured Pixtral endpoint, selecting a demo scene does not add that hazard category to the YOLO detector.")
                include_uploads = st.checkbox("Include images from the uploader in this run", value=False)
                demo_media = [DemoMedia(ROOT / "demo_samples" / demo_by_name[name]["filename"],
                                        name.replace(" ", "_"))
                              for name in selected_demo_names]
                media_items = demo_media + (list(media_items) if include_uploads else [])
if st.button("Run inspection", disabled=not media_items):
    st.session_state.pop("result", None)
    try:
        with st.spinner("Loading the detector and safety knowledge base (first run can take a few seconds)…"):
            model = detector(model_choice)
            kb = index(hybrid)
        inference_imgsz = 320 if model_choice == "Indian PPE fine-tune (4 classes)" else 640
        hazard_vision = (PixtralHazardVision(pixtral_base_url, pixtral_api_key, pixtral_model_name)
                         if pixtral_base_url else None)
        advisor = (OpenAICommentary(comment_model_name, api_key=api_key)
                   if provider == "OpenAI API" else None)
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
        completed = []
        for media_index, media in enumerate(media_items):
            start = perf_counter()
            records, previews = [], []
            suffix = Path(media.name).suffix.lower()
            with tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / ("upload" + suffix)
                target.write_bytes(media.getbuffer())
                with st.spinner(f"Inspecting {media.name} ({media_index + 1}/{len(media_items)})"):
                    for timestamp, frame in frames(target, suffix in {".mp4", ".mov", ".avi"}, interval, limit):
                        objects, annotated = detect(model, frame, imgsz=inference_imgsz)
                        ok, original = cv2.imencode(".jpg", frame)
                        if not ok:
                            raise ValueError("Could not encode frame.")
                        hazards, warnings = inspect(objects, kb, meta, taxonomy,
                                                    original.tobytes(), vision=hazard_vision,
                                                    commentary=advisor)
                        annotated = annotate_hazards(annotated, hazards)
                        records.append(dict(timestamp_seconds=timestamp, objects=objects,
                                            hazards=hazards, warnings=warnings))
                        ok, preview = cv2.imencode(".jpg", annotated)
                        if ok:
                            previews.append((timestamp, preview.tobytes()))
            output = report(meta, records, perf_counter() - start)
            output["source_media"] = media.name
            output["retrieval_mode"] = "BM25 + BGE" if hybrid else "BM25"
            output["ai_commentary_enabled"] = advisor is not None
            output["ai_commentary_provider"] = "OpenAI API" if advisor else "Off"
            output["ai_commentary_model"] = advisor.model if advisor else None
            output["hazard_vision_provider"] = "Pixtral 12B" if hazard_vision else "Off"
            output["hazard_vision_model"] = hazard_vision.model if hazard_vision else None
            output["inference_imgsz"] = inference_imgsz
            from datetime import datetime, timezone
            output["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            completed.append((output, previews, media.name))
        history = st.session_state.get("inspection_history", [])
        st.session_state["inspection_history"] = [item[0] for item in completed] + history
        st.session_state["inspection_history"] = st.session_state["inspection_history"][:20]
        st.session_state["result"] = completed
    except Exception as exc:
        st.error(f"Inspection could not run ({type(exc).__name__}). Check model, dependencies and media.")
        st.caption(str(exc) or "No additional error details were provided.")
if "result" in st.session_state:
    results = st.session_state["result"]
    if len(results) > 1:
        st.subheader(f"Batch summary · {len(results)} images")
        st.dataframe([
            {"Image": name, "Screening result": output["overall_risk_level"],
             "Identified hazards": len(output["identified_hazards"])}
            for output, _, name in results
        ], hide_index=True, use_container_width=True)
    for result_index, (output, previews, media_name) in enumerate(results):
        st.markdown(f"#### {media_name}")
        hazards = output.get("identified_hazards", [])
        detections = output.get("detected_objects", [])
        if hazards:
            high_count = sum(item.get("severity") == "High" for item in hazards)
            st.subheader(f"Potential danger detected · {high_count} high-priority risk(s)")
            st.caption("Screening result only. Confirm each observation at the site before taking action.")
        else:
            st.subheader("No mapped risk cue detected")
            if output.get("hazard_vision_provider") == "Pixtral 12B":
                st.info("Pixtral returned no supported visible hazard cue for the processed frames. This is not evidence that the site is safe; review the scene and the model’s category limits.")
            elif not detections:
                st.info("YOLO found no mapped PPE risk cue from its four trained classes. This does not show that the site is safe, and this detector does not recognize the other demo hazard categories.")
            else:
                st.info("No mapped PPE risk cue was produced from these detections. Confirm conditions at the site; this does not establish that it is safe.")
        if output.get("warnings"):
            st.warning("Some advisory processing was unavailable: " + "; ".join(output["warnings"]))
        if detections:
            detection_counts = {}
            for detection in detections:
                row = detection_counts.setdefault(detection["label"], {"Label": detection["label"], "Count": 0, "Highest confidence": 0.0})
                row["Count"] += 1
                row["Highest confidence"] = max(row["Highest confidence"], detection["confidence"])
            with st.expander(f'Model detections ({len(detections)})'):
                st.dataframe(list(detection_counts.values()), hide_index=True, use_container_width=True)
        else:
            st.caption(f'No YOLO object boxes passed the {output.get("detection_confidence_threshold", 0.25):.0%} confidence threshold in the processed media.')
        for timestamp, preview in previews:
            with st.expander(f"Detections at {timestamp:.2f} s"):
                st.image(preview)
                if output.get("hazard_vision_provider") == "Pixtral 12B":
                    st.caption("Pixtral localization boxes are approximate visual cues. Review the original frame before acting.")
        for item in hazards:
            st.markdown(f'### {item["hazard"]}')
            danger_label = "POTENTIAL DANGER — HIGH PRIORITY" if item.get("severity") == "High" else "CAUTION — MODERATE PRIORITY"
            if item.get("severity") == "High":
                st.error(danger_label)
            else:
                st.warning(danger_label)
            st.markdown(f'**Potential risk category:** {category_label(item)}')
            st.markdown(f'**Possible harm:** {item.get("potential_harm", "Confirm the possible consequence with a competent site supervisor.")}')
            cue_score = item.get("visual_risk_estimate_percent")
            if cue_score is not None:
                label = "Pixtral model score" if item.get("observation_origin") == "pixtral_vision" else "YOLO visual cue confidence"
                st.caption(f'{label}: {cue_score:.1f}% (not an accident probability or calibrated hazard probability).')
            st.markdown(f'**What the model saw:** {item["visual_evidence"]}')
            if item.get("ai_comment"):
                st.markdown("**OpenAI comment (advisory):** " + item["ai_comment"])
            st.markdown(f'**Assessment:** {assessment_label(item)}')
            st.caption(f'Why ({item.get("assessment_basis", "knowledge-base review")}): {assessment_detail(item)}')
            st.markdown(f'**Immediate action:** {item["corrective_action"]}')
            preventive = item.get("preventive_measures", [])
            if preventive:
                st.markdown("**How to prevent it**")
                for measure in preventive:
                    st.markdown(f"- {measure}")
            for source in item["retrieved_sources"]:
                st.markdown(f'**Related rule:** {source["indian_regulation"]} · {source["provision_number"]} · {source["verification_status"]}')
                st.markdown("[Open the Ministry’s Labour Codes page and source documents](https://www.labour.gov.in/offerings/schemes-and-services/details/labour-codes-gzNzQzMtQWa)")
                st.caption(source.get("required_safety_condition", ""))
            if item["model_assessment"]:
                st.caption("AI visual assessment; verify on site.")
                st.json(item["model_assessment"])
            st.divider()
        st.json(output, expanded=False)
        stem = Path(media_name).stem or f"inspection-{result_index + 1}"
        st.download_button("Download JSON", json.dumps(output, indent=2),
                           f"{stem}-inspection.json", key=f"inspection-json-{result_index}")
        try:
            pdf_data = pdf(output)
        except ImportError:
            st.caption("PDF export requires the reportlab package listed in requirements.txt.")
        else:
            st.download_button("Download PDF", pdf_data, f"{stem}-inspection.pdf",
                               "application/pdf", key=f"inspection-pdf-{result_index}")
