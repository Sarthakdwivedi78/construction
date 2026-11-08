# Missing-PPE benchmark and deployment assessment

**Dataset.** Huang and Cheng, *Dataset of Personal Protective Equipment*, version 6, CC BY 4.0. The [publisher release](https://data.mendeley.com/datasets/zkzghjvpn2/6) includes 457 validation images and no test split. The archive hash matched the publisher’s SHA-256. All 457 validation images had usable annotations and distinct file hashes. The evaluation excluded training images and checked for exact image-file overlap with that release’s training split.

**Reference labels.** A positive image has at least one published NoHelmet or NoVest box. A negative image has PPE annotations and neither of those labels. The benchmark combines construction-related and other PPE imagery. Image site identities and overlap with checkpoint training are unknown. The labels describe visible PPE and leave site-wide danger and Indian regulatory applicability undetermined.

**Protocol.** The unchanged YOLO26n checkpoint ran at 640 pixels on CPU. Scoring used the app's default 0.25 confidence threshold. A 0.01 inference floor retained scores for the rank metrics and descriptive threshold table. BM25 and the evidence gate ran on the detector outputs at the app threshold. The vision-language provider was disabled. No model or threshold was tuned on the validation split.

| Reference | Alert | No alert |
|---|---:|---:|
| Missing PPE | 200 | 13 |
| No annotated missing PPE | 117 | 127 |

| Image-level metric | Result |
|---|---:|
| Recall | 93.9% |
| Precision | 63.1% |
| Specificity | 52.0% |
| Accuracy | 71.6% |
| Balanced accuracy | 73.0% |
| F1 | 75.5% |
| ROC AUC | 84.6% |
| Average precision | 80.5% |
| Recall 95% Wilson interval | 89.8%–96.4% |
| Specificity 95% Wilson interval | 45.8%–58.2% |

Intervals are descriptive and treat images as independent. Unknown scene clustering can make them too narrow.

| Cue | Image precision | Image recall | Matched boxes at IoU ≥ 0.50 | Box precision | Box recall |
|---|---:|---:|---:|---:|---:|
| NO-Hardhat | 63.5% | 91.4% | 14 / 214 | 2.9% | 6.5% |
| NO-Safety Vest | 44.7% | 93.6% | 197 / 222 | 17.7% | 88.7% |

Boxes are matched within an image and cue class, in descending prediction confidence, with one use per annotation. Geometry can differ across annotation conventions.

| Threshold | Recall | Specificity | Alert precision |
|---:|---:|---:|---:|
| 0.15 | 94.8% | 47.1% | 61.0% |
| 0.25 | 93.9% | 52.0% | 63.1% |
| 0.4 | 90.1% | 59.4% | 66.0% |
| 0.6 | 76.5% | 77.9% | 75.1% |

CPU median detector latency was 199 ms per image and p95 was 506 ms, including rendering of the detector preview. Mean BM25 and evidence-gate time was 0.46 ms per image.

The evidence gate returned Insufficient Evidence for all 2,441 candidate findings, as expected without verified site records. No-alert images do not establish safety. The high missing-vest false-alert count and 13 image-level misses support use as a screening aid with human review.

The [ConstructionSite 10k scene benchmark](https://huggingface.co/datasets/LouisChen15/ConstructionSite) could support a separate expert-annotated construction-scene evaluation. Its protected test data returned HTTP 401 without authenticated access.
