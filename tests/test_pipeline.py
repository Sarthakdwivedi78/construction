import json
import unittest

from reasoning import inspect
from retrieval import Index, ROOT
from spatial import relationships

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.index = Index()
        self.taxonomy = json.loads((ROOT/"taxonomy.json").read_text(encoding="utf-8-sig"))
        self.objects = [{"label": "NO-Hardhat", "xyxy": [2, 3, 6, 7], "confidence": 0.9},
                        {"label": "Person", "xyxy": [0, 0, 10, 25], "confidence": 0.9}]

    def test_ppe_association(self):
        self.assertEqual(relationships(self.objects)[0]["person_detection_index"], 1)
        self.assertEqual(relationships(self.objects + self.objects[1:]), [])

    def test_jurisdiction(self):
        self.assertTrue(any(s["jurisdiction"] == "Central" for s in
                            self.index.search("toxic gas", "confined", "Central")))
        self.assertFalse(any(s["jurisdiction"] == "Central" for s in
                             self.index.search("toxic gas", "confined", "State")))

    def test_2026_central_rules_are_not_returned_for_state_authority(self):
        central = self.index.search("construction site scrap debris obstruct access", "housekeeping", "Central")
        state = self.index.search("construction site scrap debris obstruct access", "housekeeping", "State")
        self.assertIn("CR2026-30-hygiene", [row["hazard_id"] for row in central])
        self.assertNotIn("CR2026-30-hygiene", [row["hazard_id"] for row in state])
        dust = self.index.search("respirable dust fume exposure", "dust", "Central")
        self.assertIn("CR2026-32(i)", [row["hazard_id"] for row in dust])

    def test_2026_lighting_and_emergency_rules_are_central_only(self):
        lighting = self.index.search("construction access lighting portable emergency lamps",
                                     "access", "Central")
        drills = self.index.search("quarterly construction emergency preparedness mock drill",
                                   "emergency", "Central")
        self.assertEqual(lighting[0]["hazard_id"], "CR2026-46-lighting")
        self.assertEqual(drills[0]["hazard_id"], "CR2026-59-emergency-drills")
        self.assertNotIn("CR2026-46-lighting", [row["hazard_id"] for row in
                                                 self.index.search("construction lighting", "access", "State")])
        self.assertNotIn("CR2026-59-emergency-drills", [row["hazard_id"] for row in
                                                        self.index.search("quarterly mock drills", "emergency", "State")])

    def test_invalid_model_citation_abstains(self):
        class InvalidVision:
            def ask(self, instructions, data, jpeg):
                return {"hazards": [], "status": "Violation",
                        "reasoning": "Unsupported", "cited_id": "invented"}
        hazards, warnings = inspect(self.objects, self.index, {}, self.taxonomy, vision=InvalidVision())
        self.assertEqual(hazards[0]["violation_status"], "Insufficient Evidence")
        self.assertIsNone(hazards[0]["model_assessment"])
        self.assertTrue(warnings)

    def test_reviewed_decisions_require_matching_evidence(self):
        import hashlib
        class FixtureVision:
            def ask(self, instructions, data, jpeg):
                return {"hazards": [{"category": "confined", "hazard": "Confined entry",
                                     "visual_evidence": "Worker at hatch", "query": "toxic gas"}],
                        "status": "Violation", "reasoning": "Candidate",
                        "cited_id": "CR2026-31(iii)"}
        jpeg = b"synthetic frame"
        check = {"frame_sha256": hashlib.sha256(jpeg).hexdigest(), "reviewer": "fixture reviewer",
                 "applicability_confirmed": True, "hazard": "Confined entry",
                 "visual_evidence": "Worker at hatch", "evidence": "Reviewed fixture record",
                 "inspection_date": "2026-11-07"}
        meta = {"authority": "Central", "inspection_date": "2026-11-07",
                "reviewed_checks": {"CR2026-31(iii)": check}}
        for status in ("Violation", "No Violation"):
            check["status"] = status
            result, _ = inspect([], self.index, meta, self.taxonomy, jpeg, FixtureVision())
            self.assertEqual(result[0]["violation_status"], status)
        result, _ = inspect([], self.index, meta, self.taxonomy, b"different frame", FixtureVision())
        self.assertEqual(result[0]["violation_status"], "Insufficient Evidence")

    def test_video_summary_merges_only_overlapping_cues(self):
        from reasoning import summarize_hazards
        cue = {"hazard": "Possible missing hardhat", "violation_status": "Insufficient Evidence",
               "evidence_box": [0, 0, 20, 20]}
        frames = [{"timestamp_seconds": 0, "hazards": [cue]},
                  {"timestamp_seconds": 2, "hazards": [dict(cue, evidence_box=[1, 0, 21, 20]),
                                                       dict(cue, evidence_box=[30, 30, 50, 50])]}]
        summary = summarize_hazards(frames)
        self.assertEqual([h["observations"] for h in summary], [2, 1])
        self.assertEqual(summary[0]["last_seen_seconds"], 2)

    def test_inspection_history_persists_reports_without_media_payload(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from history_store import load_history, save_inspection
        with TemporaryDirectory() as folder:
            database = Path(folder) / "history.sqlite3"
            report = {"inspection_id": "fixture-1", "site_type": "Test site",
                      "overall_risk_level": "Review required", "frames_processed": 1}
            saved = save_inspection(report, database)
            self.assertTrue(saved["created_at"])
            save_inspection(report, database)
            loaded = load_history(database_path=database)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["inspection_id"], "fixture-1")
            self.assertNotIn("image_bytes", loaded[0])

    def test_ablation_keeps_ungated_and_reviewed_decisions_separate(self):
        from run_ablation import choose_prediction
        finding = {"category": "ppe", "model_assessment": {"status": "Violation", "cited_id": "OSH6-ppe"},
                   "decision_code": "INSUFFICIENT_EVIDENCE", "selected_provision_id": "OSH6-ppe"}
        self.assertEqual(choose_prediction("fixture", "ppe", [finding], False, [])["status"], "VIOLATION")
        self.assertEqual(choose_prediction("fixture", "ppe", [finding], True, [])["status"],
                         "INSUFFICIENT_EVIDENCE")

    def test_tuning_rejects_shared_validation_and_test_queries(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from tune_retrieval import tune
        with TemporaryDirectory() as folder:
            validation = Path(folder) / "validation.jsonl"
            test = Path(folder) / "test.jsonl"
            row = {"reviewed": True, "query": "Head protection", "category": "ppe",
                   "relevant_ids": ["OSH6-ppe"]}
            validation.write_text(json.dumps(dict(row, case_id="v")), encoding="utf-8")
            test.write_text(json.dumps(dict(row, case_id="t", query="head   protection")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identical normalized queries"):
                tune(self.index, validation, test, 3)

    def test_reranker_only_receives_filtered_candidates(self):
        class FakeReranker:
            def predict(self, pairs):
                if any("Rule 31(iii)" in text for _, text in pairs):
                    raise AssertionError("Central rule leaked into a state-only query")
                return [0.0] * len(pairs)
        self.index.reranker = FakeReranker()
        result = self.index.search("toxic gas", "confined", "State", state="West Bengal")
        self.assertEqual([row["hazard_id"] for row in result], ["OSH6-confined"])

    def test_local_vision_serializes_image_and_rejects_nonobjects(self):
        from io import BytesIO
        from local_vision import LocalVision
        from unittest.mock import patch
        adapter = LocalVision("fixture")
        with patch.object(adapter.opener, "open") as send:
            send.return_value = BytesIO(b'{"message":{"content":"{\\"hazards\\":[]}"}}')
            self.assertEqual(adapter.ask("inspect", {"boxes": []}, b"abc"), {"hazards": []})
            request = send.call_args.args[0]
            body = json.loads(request.data)
            self.assertTrue(request.full_url.startswith("http://127.0.0.1:"))
            self.assertEqual(body["messages"][1]["images"], ["YWJj"])
            send.return_value = BytesIO(b'{"message":{"content":"[]"}}')
            with self.assertRaisesRegex(ValueError, "JSON object"):
                adapter.ask("inspect", {}, b"abc")

    def test_review_records_stay_out_of_both_vision_calls(self):
        seen = []
        class RecordingVision:
            def ask(self, instructions, data, jpeg):
                seen.append(data)
                return {"hazards": []} if "categories" in data else {
                    "status": "Insufficient Evidence", "reasoning": "Unknown", "cited_id": None}
        inspect(self.objects, self.index, {"reviewed_checks": {"private": {}}},
                self.taxonomy, b"fixture", RecordingVision())
        self.assertEqual(len(seen), 2)
        self.assertTrue(all("reviewed_checks" not in call["site"] for call in seen))

    def test_site_measurements_require_numeric_value_and_provenance(self):
        from measurements import parse_measurements
        sample = {"name": "trench depth", "unit": "m", "value": 1.7, "method": "tape measure",
                  "source_reference": "fixture record 1"}
        self.assertEqual(parse_measurements(json.dumps([sample])), [sample])
        for bad in (dict(sample, value=True), dict(sample, value=float("inf")),
                    dict(sample, source_reference=" ")):
            with self.assertRaises(ValueError):
                parse_measurements(json.dumps([bad]))

    def test_machine_person_overlap_is_only_an_image_plane_cue(self):
        detections = [{"label": "Person", "xyxy": [0, 0, 10, 10]},
                      {"label": "machinery", "xyxy": [8, 8, 30, 30]},
                      {"label": "vehicle", "xyxy": [10, 0, 30, 10]}]
        rows = relationships(detections)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["equipment_detection_index"], 1)
        self.assertIn("physical distance unknown", rows[0]["relation"])

    def test_earlier_inspection_excludes_later_provisions(self):
        self.assertEqual(self.index.search("PPE", "ppe", inspection_date="2023-03-20"), [])
        current = self.index.search("PPE", "ppe", inspection_date="2026-11-07")
        self.assertEqual(current[0]["hazard_id"], "OSH6-ppe")

    def test_dataset_draft_rejects_detection_report_for_other_image(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from draft_review import draft
        import hashlib
        with TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "fixture.jpg"
            image.write_bytes(b"source fixture")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([{"case_id": "sample", "image": image.name,
                                             "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]))
            self.assertFalse(list(draft(manifest))[0]["reviewed"])
            (root / "sample.json").write_text(json.dumps({"media_sha256": "incorrect"}))
            with self.assertRaisesRegex(ValueError, "Report does not match"):
                list(draft(manifest, root))

    def test_lora_preflight_rejects_groups_that_cross_splits(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from train_vision_lora import load_splits
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "validation", "test"):
                (root / (split + ".jpg")).write_bytes(split.encode())
                row = {"case_id": split, "reviewed": True, "reviewer": "test fixture",
                       "site_id": split, "site_group": "shared project", "image": split + ".jpg",
                       "question": "What can be established?", "retrieved_regulations": [],
                       "answer": {"violation_status": "INSUFFICIENT_EVIDENCE",
                                  "hazard": "Fixture concern", "reasoning": "Fixture", "corrective_action": "Verify evidence"}}
                (root / (split + ".jsonl")).write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cross-split"):
                load_splits(root, root)

    def test_ablation_can_read_prepared_heldout_split(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from run_ablation import load_references
        with TemporaryDirectory() as folder:
            path = Path(folder) / "test.jsonl"
            row = {"case_id": "example", "reviewed": True, "hazard_category": "ppe",
                   "answer": {"violation_status": "INSUFFICIENT_EVIDENCE"},
                   "retrieved_regulations": ["OSH6-ppe"]}
            path.write_text(json.dumps(row), encoding="utf-8")
            result = load_references(path, self.taxonomy)["example"]
            self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")
            self.assertEqual(result["acceptable_provision_ids"], ["OSH6-ppe"])

    def test_ablation_passes_measurement_context_without_ground_truth(self):
        from run_ablation import prediction_context
        measurement = {"name": "trench depth", "unit": "m", "method": "survey",
                       "source_reference": "record 1", "value": 1.7}
        case = {"state": "West Bengal", "site_measurements": [measurement],
                "answer": {"violation_status": "VIOLATION"}, "reviewed_checks": {"private": {}},
                "acceptable_provision_ids": ["reference"]}
        context = prediction_context(case)
        self.assertEqual(context["site_measurements"], [measurement])
        self.assertFalse({"answer", "reviewed_checks", "acceptable_provision_ids"} & context.keys())

    def test_ocr_compact_headings_keep_provenance(self):
        from ingest import chunk_pages
        rows = list(chunk_pages([(1, "CHAPTERXIII\n127.Safeaccess\nOCR text fixture."),
                                 (2, "128.Trenchprotection\nSecond fixture.")],
                                "https://example.test/fixture", "Fixture", "Historical Central", {1, 2}))
        self.assertEqual([r["provision_reference_candidate"] for r in rows], ["127", "128"])
        self.assertEqual(rows[0]["chapter_candidate"], "CHAPTER XIII")
        self.assertEqual(rows[0]["ocr_pages"], [1])
        self.assertEqual(rows[1]["ocr_pages"], [2])
        self.assertIn("compare_ocr_with_source", rows[0]["quality_flags"])
