# Universal Surgical Alignment (USA)
### Multi-Level Video-Language Matching with Action-Driven Digital Twins

**A Refined Thesis & Research Paper Proposal (v2)**
Grounded in Surgical Data Science, Computer Vision, and Multimodal Deep Learning

---

## Executive Summary

This document outlines a refined research proposal and master execution plan for a master's thesis. Based on the latest slide iteration, the project's scope has been sharpened: we have eliminated the preliminary task of developing and compiling canonical text descriptions from raw transcripts. Instead, we treat canonical surgical checklists as predefined, standard procedural references (e.g., standard clinical protocols or gold-standard annotations from datasets like CAT-SG, StrasBypass70, and PSI-AVA).

The entire research now focuses strictly on the core challenge: **hierarchical video-language alignment**. The proposed framework, **Universal Surgical Alignment (USA)**, maps streaming raw surgical videos to these standard canonical steps in a joint embedding space. By serializing visual interactions as structured, spatio-temporal **Action-Driven Digital Twins (ActDTs)** (Subject-Action-Object-Location-Depth), the model performs open-vocabulary, zero-shot surgical workflow recognition. Crucially, the system incorporates step-aware feature representations and temporal rectification mechanisms to handle skipped, repeated, or out-of-order steps, establishing a robust clinical progress tracker and goal-verification checkpoint system.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Problem Statement & Literature Gap](#2-problem-statement--literature-gap)
3. [Refined Research Question & Hypothesis](#3-refined-research-question--hypothesis)
4. [Technical Methodology & Architecture](#4-technical-methodology--architecture)
5. [Comprehensive Answers to Core Research Questions](#5-comprehensive-answers-to-core-research-questions)
6. [Required Data & Material Requirements](#6-required-data--material-requirements)
7. [Master Implementation Timeline](#7-master-implementation-timeline-beginning-to-end)
8. [Microscopic Development Roadmap & Task Checklist](#8-microscopic-development-roadmap--task-checklist)

---

## 1. Abstract

Computer-assisted surgical (CAS) systems require a granular and explainable understanding of surgical workflows to monitor clinical procedures, provide safety checkpoints, and alert teams to skipped steps. Current state-of-the-art models are bottlenecked by supervised, center-specific training; a model trained on one hospital's annotated frames degrades significantly when transferred to another medical center due to center-specific procedural variations.

To solve this generalization barrier, we propose **Universal Surgical Alignment (USA)**, a framework that aligns streaming surgical video directly with standard canonical steps using hierarchical video-language pretraining. We represent raw video clips as structured, spatio-temporal Action-Driven Digital Twins (ActDTs) serialized in JSON (Subject-Action-Object-Location-Depth). Rather than performing rigid frame-by-frame categorization, we use a shared visual-linguistic text encoder to align these dynamic digital twins with multi-level canonical step checklists (**Phase → Step → Action → Tool → Anatomy**) in a joint embedding space.

To handle procedural anomalies, we integrate step-aware feature prototypes to pad skipped steps, alongside a rectification mechanism that uses mirroring and center-duplication to extend ongoing streams into pseudo-complete sequences, enabling continuous temporal refinement. We benchmark our zero-shot generalization capabilities across three surgical specialties (ophthalmic, abdominal, and urological robotic surgeries) and demonstrate high robustness to multi-center workflow variations.

---

## 2. Problem Statement & Literature Gap

Surgical workflow recognition is vital for clinical safety, yet existing computer vision models face severe limitations that the proposed thesis directly addresses:

- **Rigid Categorical Hardcoding** — Traditional models (e.g., PhaseNet, SV-RCNet, OperA) map video frames to a fixed, closed-set classification head. This prevents open-vocabulary generalization; adapting a network to a new surgical specialty requires re-annotating and retraining the entire classifier from scratch.
- **The Multi-Center Failure Mode** — When evaluated across clinics, sequential models collapse due to workflow deviations. For instance, zero-shot models in HecVL pre-trained on Strasbourg University Hospital protocols (StrasBypass70) experienced significant performance drops when evaluated on Bern University Hospital data (BernBypass70) because Bern surgeons routinely performed different step sequences or skipped steps.
- **Semantic Ambiguity of Flat Spaces** — Projecting multiple task granularities (e.g., broad phases vs. fine-grained tool actions) into a single, unified embedding space introduces semantic "blurring." Short-term visual clips representing brief actions are pulled toward broad textual summaries, confusing the model's temporal understanding.
- **Visual Near-Identity in the Surgical Field** — Consecutive clips within the same surgery are visually almost identical — they feature the same patient, lighting, and general tools, differing only in subtle tool-tissue interactions. Standard video-language retrieval models (e.g., CLIP4Clip, X-CLIP) fail because they optimize for global visual appearance rather than localized, action-level transitions.
- **Temporal Blindness & Jitter** — Most online models classify frames independently or use brief historical windows, leading to fragmented, physically inconsistent predictions (e.g., predicting an incision step after implant placement). They lack a global, chronological workflow perspective.

---

## 3. Refined Research Question & Hypothesis

**Primary Research Question (RQ):**
Can we leverage hierarchical video-language alignment to automatically synchronize raw surgical video segments with pre-existing, canonical step checklists to enable zero-shot procedural recognition and maintain tracking continuity when steps are skipped or performed out of sequence?

**Thesis Hypothesis (H):**
By converting raw clinical video streams into structured, text-serialized Action-Driven Digital Twins (ActDTs) and performing contrastive learning across separate, hierarchical visual-textual embedding spaces, we can align visually near-identical surgical clips to their correct chronological position on a standard canonical checklist. This alignment remains robust to procedural variations (skips, substitutions) by utilizing step-aware padding and global temporal rectification.

---

## 4. Technical Methodology & Architecture

The revised USA framework consists of two core pipelines: the **Action-Driven Video Parser** and the **Spatio-Temporal Hierarchical Aligner**.

```
                       [ Streaming Surgical Video ]
                                   │
                                   ▼
                     [ Action-Driven Video Parser ]
             (Mask2Former + Qwen3-VL + DepthAnythingV3)
                                   │
                                   ▼
                     [ Structured ActDT (JSON) ]
                   "Subj-Act-Obj-Loc-CentroidDepth"
                                   │
                                   ▼
                   [ Spatio-Temporal Hierarchical Aligner ]
               (HecVL Separated Multi-Space Contrastive)
                                ▲     ▲
                                │     │  (ICL Loss / HRAM)
                                │     ▼
[ Predefined Canonical ] ───────┘   [ Progress & Safety Checkpoints ]
   Checklists (MSSU)                "Skipped Step Warnings & Goals"
```

### A. Forward Video Parsing & ActDT Construction (Visual-to-Textual Bridge)

To analyze raw clinical video, we develop a forward-parsing pipeline that translates high-dimensional pixel streams into structured, text-serialized Action-Driven Digital Twins (ActDTs). This avoids visual token compression and allows direct reasoning:

1. **Spatio-Temporal Segmentation** — A custom Mask2Former model with a VideoSwin backbone (pretrained on surgical videos via the VALOR/Watch&Learn frameworks) generates pixel-precise segmentations of 29 surgical objects and anatomies.
2. **3D Centroid Coordinate Extraction** — Instance segmentation masks are combined with DepthAnythingV3 to predict the depth of the centroid of each segmented element, mapping coordinates to 3D space (ℝ⁵: bounding box x₁, y₁, x₂, y₂ and centroid depth z).
3. **Action Attribute Generation** — Localized visual regions of interest are passed to a fine-tuned vision-language model (Qwen3-VL-8B) to generate natural-language action attributes representing tool-tissue interactions.
4. **JSON Serialization** — The output is serialized frame-by-frame as an ActDT sequence:

```json
{
  "interval": [t_start, t_end],
  "actions": [
    {"subject": "Primary Knife", "action": "Cut", "object": "Cornea", "loc_3D": [x1, y1, x2, y2, z]}
  ]
}
```

### B. Spatio-Temporal Hierarchical Aligner

Once video ActDTs are extracted, we align them to predefined canonical checklists (derived from CaDIS, CAT-SG, StrasBypass70, or PSI-AVA):

- **Hierarchical Relation Aggregation (HRAM)** — A cross-attention module that learns semantic relations T_(i↔j) across task hierarchies (Phase ↔ Step ↔ Action), allowing fine-grained tool-action features to directly assist in high-level step localization.
- **Disentangled Multi-Space Alignment** — Following HecVL, we construct separate embedding projection heads for clip-level (S_narration), step-level (S_concept), and overall procedure-level (S_abstract) embeddings. This prevents semantic blurring and preserves short-term action dynamics.
- **Inter-Task Contrastive Learning (ICL)** — Visual-textual encoders are optimized using a multi-task InfoNCE contrastive loss (L_cij) that pulls consistent task pairs (Phase ↔ Step and Action ↔ Instrument) closer together in the embedding space.

### C. Checkpoint Progress Tracking & "Skipped Step" Robustness

The primary clinical utility of this alignment is checking whether surgical goals are being hit and flagging bypassed actions. We address procedural anomalies through two key mechanisms:

- **Step-aware Feature Padding (SFR from StepAL)** — For an unlabeled video, clip embeddings are organized according to predicted steps using pseudo-labels. If a canonical step S_k is omitted by the surgeon, its corresponding slot in the feature vector is padded with the video's global average feature. This maintains dimensional consistency and ensures global temporal context is preserved for clustering and representation.
- **Temporal Rectification & Mirroring (SurgPLAN++)** — For online streaming, we apply data augmentation (mirroring, center-duplication, and down-sampling) to extend the ongoing stream into a pseudo-complete sequence. The Phase Localization Network can then predict phase segments across the entire video. If a step is skipped, the Rectification Mechanism continuously refines preceding predictions at each online step, updating the dynamic result sequence R_phase and correcting temporal anomalies based on global temporal proposals rather than frame-by-frame classification.

---

## 5. Comprehensive Answers to Core Research Questions

### Question 1: How can we fit this model to other forms of surgery?

The USA architecture is uniquely suited for cross-specialty and cross-procedure transfer because it is entirely decoupled from rigid, category-specific classifiers.

- **Open-Vocabulary Checklists** — No changes are made to the neural network. We simply substitute the text input file with the new surgery's checklist mapped to the standard hierarchy (e.g., Robotic Prostatectomy in PSI-AVA or Laparoscopic Cholecystectomy in Cholec80).
- **Cross-Procedural Foundations** — The visual and text encoders are pre-trained on massive, diverse medical video-text datasets (such as MedVideoCap-55K or SurgVLP/SVL-25K), enabling zero-shot generalization to completely unseen surgeries.
- **Generic Vision Frontends** — The forward video-parsing pipeline utilizes open-set tracking models (SAM-3, DepthAnythingV3, and Qwen3-VL) that extract spatial coordinates and interactions, serialized into standardized JSON format so the downstream aligner can match observed tool-tissue interactions to any surgical protocol.

### Question 2: How can we come up with evaluation metrics to justify this alignment?

We propose a three-tier evaluation suite to quantify performance, safety, and anomaly robustness.

**1. Zero-Shot Recognition Benchmarks** (evaluated without any fine-tuning)
- Frame-wise Accuracy & F1-Score — measures classification correctness
- Jaccard Index (IoU) — evaluates correctness of temporal segment boundaries
- Recall@K (R@1, R@5, R@10) — measures accuracy of reasoning-based text-to-video clip retrieval

**2. Process Integrity & Safety Checklist Metrics**
- **Alignment Discrepancy Index (ADI)** — average temporal offset (in seconds) between ground-truth step boundaries and aligned canonical step windows:

  ```
  ADI = (1/M) * Σ [ |t_start_GT,i − t_start_Aligned,i| + |t_end_GT,i − t_end_Aligned,i| ]
  ```

- **Sequence Order Integrity (SOI)** — evaluates whether the predicted alignment maintains chronological order, computed using Kendall's Rank Correlation (τ) or Normalized Edit Distance against the canonical checklist
- **Checkpoint Detection Recall (CDR)** — measures how accurately the model flags reached clinical milestones or safety checkpoints

**3. Anomaly & Skip Robustness Metrics**
- Skip Detection Recall (SDR) — percentage of skipped steps correctly flagged by the model
- False Alarm Rate (FAR) — percentage of valid, out-of-order variations incorrectly flagged as skipped steps

### Question 3: How are current papers unable to cover this idea?

Existing papers in the surgical AI domain hit boundaries because they solve only isolated aspects of the workflow:

| Paper | Limitation |
|---|---|
| **HecVL** | Introduces hierarchical video-language pre-training but relies on unstructured, flat textual prompts. Lacks a forward-facing video parser to convert pixel streams into structured digital twins and fails to handle center-specific sequential variations or skipped steps. |
| **MSSU (HCT/HCT+)** | Successfully establishes the multi-level hierarchy but is fully supervised, requiring massive manual annotations for every dataset, which prevents zero-shot transfer. |
| **OR3 / ActDT** | Solves only the inverse problem — using language queries to retrieve historical clips from a database. Does not provide a forward, online streaming parser that dynamically maps video to a canonical checklist. |
| **StepAL** | Addresses active learning and introduces step-aware representations, but is limited to frame/clip-level dataset selections and does not perform cross-modal video-text alignment. |
| **SurgPLAN++** | Incorporates temporal phase localization and rectification but operates entirely within a single-modality framework (vision only) and cannot leverage natural language prompts to perform open-vocabulary reasoning. |

---

## 6. Required Data & Material Requirements

| Dataset | Modality / Specs | Annotation Details | Role in Proposed Work |
|---|---|---|---|
| **CAT-SG / CaDIS** | Cataract microscopy (164K frames @ 5 fps) | 29 instrument/anatomy classes, 1.8M tool-tissue relations, nucleus-breaking techniques | Benchmarking the forward-parsing pipeline; validating fine-grained cataract step alignment |
| **Cataract-1k / Cataract-101** | Cataract videos (1 fps) | 13 and 10 chronological surgical steps respectively | Evaluating step-aware feature representations and temporal sequence consistency |
| **StrasBypass70 & BernBypass70** | Laparoscopic Gastric Bypass (Strasbourg & Bern centers) | Standard phase and step labels | Main benchmark for multi-center generalization and testing robustness to skipped/altered steps |
| **PSI-AVA** | Robotic Prostatectomy (Da Vinci, 1 fps) | 11 phases, 21 steps, 16 actions, 7 instruments | Demonstrating cross-specialty transferability (abdominal/robotic surgery) |
| **SurgVLP / SVL** | Multi-procedure lecture videos | 25K clip-level narrations, 10K phase concepts, 1K video abstracts | Pretraining visual and textual encoders (ResNet-50 / VideoSwin and BioClinicalBert) |
| **MedVideoCap-55K** | Diverse clinical, imaging, and animated clips | Detailed textual captions | Large-scale pretraining to maximize zero-shot generalization capability |

---

## 7. Master Implementation Timeline (Beginning to End)

The timeline is optimized into a 12-month schedule focused entirely on development, alignment modeling, and cross-center evaluation:

```
Month 1-2 ─────► Month 3-4 ─────► Month 5-6 ─────► Month 7-8 ─────► Month 9-10 ────► Month 11-12
[Data Prep]      [Parser Dev]     [Aligner Dev]    [Robustness]     [Benchmark]      [Writing]
SurgVLP, CAT-SG  Segment, Depth   HCT+ & HecVL     SFR & SurgPLAN   Evaluation       Drafting &
& Bypass Preps   ActDT JSONs      Multi-Spaces     Skip-Mitigation  Multi-Center     Review Cycle
```

- **Months 1–2: Data Ingestion & Environment Setup** — Set up video-language backbones, prepare SurgVLP and MedVideoCap-55K pretraining pipelines, and preprocess CAT-SG, Bypass, and PSI-AVA.
- **Months 3–4: Action-Driven Video Parser Development** — Build the forward-facing segmentation (Mask2Former), centroid coordinates/depth extractor (DepthAnythingV3), action classifier (Qwen3-VL-8B), and serialize as JSON ActDTs.
- **Months 5–6: Spatio-Temporal Hierarchical Aligner Development** — Implement HRAM cross-attention and HecVL multi-space contrastive learning. Train visual and text encoders.
- **Months 7–8: Robustness & Temporal Rectification Modeling** — Implement Step-aware Feature Representation (SFR) padding from StepAL and the SurgPLAN++ mirroring, center-duplication, and dynamic results rectification.
- **Months 9–10: Benchmarking & Multi-Center Generalization** — Run zero-shot benchmarks across cataract, gastric bypass (StrasBypass70 vs. BernBypass70), and prostatectomy. Extract ADI, SOI, and SDR metrics.
- **Months 11–12: Writing & Publication Pipeline** — Draft the paper sections and complete review cycles for submission.

---

## 8. Microscopic Development Roadmap & Task Checklist

### A. Development Tasks

- [ ] **D1** — Ingest and preprocess SurgVLP, CAT-SG, StrasBypass70, BernBypass70, and PSI-AVA video frames at 1 fps.
- [ ] **D2** — Set up the video-language backbone environment. Download pretrained checkpoints of ResNet-50, VideoSwin, BioClinicalBert, and Qwen3-VL-8B.
- [ ] **D3** — Develop the forward-facing segmentation script using Mask2Former and Swin Small on the CaDIS dataset to output instrument masks.
- [ ] **D4** — Integrate DepthAnythingV3 to extract depth coordinates from segmented centroids and combine them with the bounding boxes.
- [ ] **D5** — Build the Qwen3-VL-8B inference routine to query tool-tissue interaction actions and serialize results as ActDT JSON files.
- [ ] **D6** — Program the Hierarchical Relation Aggregation Module (HRAM) with cross-attention blocks to link Phase, Step, and Action features.
- [ ] **D7** — Implement the HecVL fine-to-coarse projection heads, constructing the separate narration, concept, and abstract embedding spaces.
- [ ] **D8** — Code the Inter-Task Contrastive Learning (ICL) loss functions and optimize using an alternating training pipeline.
- [ ] **D9** — Write the Step-aware Feature Representation (SFR) vector extraction script, including feature padding for missing step slots.
- [ ] **D10** — Develop the SurgPLAN++ data augmentation module (mirroring, center-duplication, down-sampling) for temporal phase localization.
- [ ] **D11** — Build the dynamic results rectification loop to continuously update online predictions based on retrospective phase proposals.
- [ ] **D12** — Script the evaluation metrics pipeline to compute Accuracy, F1, Jaccard Index, ADI, SOI, CDR, and Recall@K.

### B. Writing Tasks

- [ ] **W1** — Draft the Introduction section, framing the supervised, rigid classification bottleneck and highlighting the multi-center clinical failure mode.
- [ ] **W2** — Write the Related Work section, summarizing and identifying limitations in HecVL, MSSU/HCT, OR3, StepAL, and SurgPLAN++.
- [ ] **W3** — Author the Methodology (Phase I & II) section, detailing the mathematical representation of the ActDT parser.
- [ ] **W4** — Write the Methodology (Phase III & IV) section, detailing HRAM cross-attention, multi-space contrastive loss, and temporal rectification equations.
- [ ] **W5** — Compile the Experimental Setup section, documenting training parameters (A100 GPU settings, learning rates, batch sizes, optimizer choices).
- [ ] **W6** — Create the Results – Zero-Shot Recognition tables, demonstrating cross-procedure performance improvements on Cataract, Hysterectomy, and Prostatectomy datasets.
- [ ] **W7** — Draft the Results – Multi-Center Robustness section, comparing StrasBypass70 vs. BernBypass70 alignment performance and highlighting skip-mitigation success.
- [ ] **W8** — Formulate the Ablation Studies tables and narrative, demonstrating the quantitative impact of HRAM, ICL, SFR, and temporal augmentation.
- [ ] **W9** — Draft the Discussion & Conclusion sections, analyzing explainability through GNNExplainer/visualizations, clinical checklists, and submitting the manuscript for peer review.