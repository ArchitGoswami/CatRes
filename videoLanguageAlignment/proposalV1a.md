# Universal Surgical Alignment (USA): Multi-Level Video-Language Matching with Action-Driven Digital Twins

## A Comprehensive Master Thesis Proposal & Execution Plan

---

# PART I: RESEARCH FOUNDATION

## 1. Executive Summary

This master thesis proposes **Universal Surgical Alignment (USA)**, a framework that aligns streaming surgical video directly with predefined canonical step checklists using hierarchical video-language pretraining. The system converts raw video into structured **Action-Driven Digital Twins (ActDTs)**—JSON-serialized representations of tool-tissue interactions (Subject-Action-Object-Location-Depth)—and aligns them to multi-level canonical references (Phase → Step → Action → Instrument → Anatomy) in disentangled embedding spaces.

**Key Scope Clarification**: We treat canonical surgical checklists as predefined procedural references (from clinical protocols or gold-standard dataset annotations), eliminating the preliminary task of generating canonical descriptions from raw transcripts. The entire research focuses on the **core alignment challenge**.

**Core Contributions**:
1. A forward-facing video parser that converts pixel streams into structured ActDT representations
2. Hierarchical contrastive alignment across disentangled embedding spaces
3. Robust handling of skipped/reordered steps via step-aware padding and temporal rectification
4. Zero-shot generalization across surgical specialties without re-annotation

---

## 2. Problem Statement

### 2.1 The Clinical Need

Computer-assisted surgical (CAS) systems require granular, explainable workflow understanding to:
- Monitor clinical procedures in real-time
- Provide safety checkpoints and goal verification
- Alert surgical teams to skipped or misordered steps
- Enable standardized training and quality assurance

### 2.2 Five Critical Technical Limitations

| Limitation | Description | Evidence |
|------------|-------------|----------|
| **Categorical Hardcoding** | Traditional models (PhaseNet, SV-RCNet, OperA) use fixed classification heads, preventing open-vocabulary generalization | Adapting to new surgery requires complete re-annotation and retraining |
| **Multi-Center Failure Mode** | Models collapse when workflows deviate between hospitals | HecVL: Significant performance drop from Strasbourg → Bern due to different step sequences |
| **Semantic Granularity Collapse** | Single embedding spaces confuse phase-level and action-level information | HecVL ablations: "Flat" single-space models perform inconsistently |
| **Visual Near-Identity Problem** | Consecutive surgical clips are visually almost identical | OR3: CLIP4Clip achieves <16% R@1 on surgical retrieval |
| **Temporal Blindness** | Frame-by-frame models lack global procedural context | SurgPLAN++: Independent classification leads to physically inconsistent predictions |

### 2.3 The Literature Gap

| Paper | Strengths | Critical Gaps |
|-------|-----------|---------------|
| **HecVL** | Hierarchical video-language pretraining | No forward video parser; no ActDT structure; fails on center-specific variations |
| **HCT (MSSU)** | Multi-level semantic hierarchy | Fully supervised; requires massive manual annotations; no zero-shot capability |
| **OR3 / ActDT** | Structured digital twin representations | Inverse direction only (text→video retrieval); no forward online parsing |
| **StepAL** | Step-aware temporal representations | Frame/clip-level only; no cross-modal video-text alignment |
| **SurgPLAN++** | Temporal rectification for online inference | Vision-only; no natural language grounding; no open-vocabulary reasoning |

**Your Thesis Fills the Gap**: No existing paper provides a unified, forward-facing pipeline that converts streaming video into structured ActDTs and aligns them to canonical checklists for real-time, cross-procedural workflow verification.

---

## 3. Research Questions & Hypothesis

### 3.1 Primary Research Question

> *Can we leverage hierarchical video-language alignment to automatically synchronize raw surgical video segments with pre-existing canonical step checklists to enable zero-shot procedural recognition and maintain tracking continuity when steps are skipped or performed out of sequence?*

### 3.2 Sub-Questions

| ID | Sub-Question | Addressed By |
|----|--------------|--------------|
| SQ1 | What is the optimal structured format for representing surgical video content? | ActDT JSON serialization |
| SQ2 | How do we maintain alignment when steps are skipped or reordered? | SFR padding + temporal rectification |
| SQ3 | How do we transfer to new surgical specialties without re-training? | Open-vocabulary text alignment |
| SQ4 | What metrics justify the alignment quality and clinical utility? | Three-tier evaluation framework |

### 3.3 Thesis Hypothesis

> By converting raw clinical video streams into structured, text-serialized Action-Driven Digital Twins (ActDTs) and performing contrastive learning across separate, hierarchical visual-textual embedding spaces, we can align visually near-identical surgical clips to their correct chronological position on a standard canonical checklist. This alignment remains robust to procedural variations (skips, substitutions) by utilizing step-aware padding and global temporal rectification.

---

# PART II: TECHNICAL METHODOLOGY

## 4. System Architecture

### 4.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CANONICAL REFERENCE LAYER                           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                    │
│  │ Phase-Level  │   │  Step-Level  │   │ Action-Level │                    │
│  │    Texts     │   │    Texts     │   │   Triplets   │                    │
│  │ (Abstracts)  │   │  (Concepts)  │   │ (Narrations) │                    │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                    │
│         └──────────────────┼──────────────────┘                            │
│                            ▼                                                │
│              Hierarchical Text Encoder (BioClinicalBERT)                    │
│                            │                                                │
│         ┌──────────────────┼──────────────────┐                            │
│         ▼                  ▼                  ▼                             │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐                         │
│   │ S_abstract│      │ S_concept │      │S_narration│  ← Disentangled      │
│   │  Space   │      │   Space   │      │   Space   │    Embedding Spaces   │
│   └────┬─────┘      └────┬─────┘      └────┬─────┘                         │
└────────┼─────────────────┼─────────────────┼────────────────────────────────┘
         │                 │                 │
         │    ┌────────────┴────────────┐    │
         │    │   CONTRASTIVE ALIGNMENT │    │
         │    │      (InfoNCE + ICL)    │    │
         │    └────────────┬────────────┘    │
         │                 │                 │
┌────────┼─────────────────┼─────────────────┼────────────────────────────────┐
│        ▼                 ▼                 ▼                                │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐                         │
│   │ V_phase  │      │ V_step   │      │ V_action │  ← Visual Projections   │
│   │ Features │      │ Features │      │ Features │                         │
│   └────┬─────┘      └────┬─────┘      └────┬─────┘                         │
│        └─────────────────┼─────────────────┘                               │
│                          ▲                                                  │
│              Hierarchical Relation Aggregation (HRAM)                       │
│                          │                                                  │
│         ┌────────────────┼────────────────┐                                │
│         │                │                │                                 │
│   ┌─────┴─────┐   ┌──────┴──────┐   ┌─────┴─────┐                          │
│   │ Temporal  │   │   ActDT     │   │   SFR     │                          │
│   │Aggregation│   │   Parser    │   │  Padding  │                          │
│   └─────┬─────┘   └──────┬──────┘   └─────┬─────┘                          │
│         └────────────────┼────────────────┘                                │
│                          ▲                                                  │
│                          │                                                  │
│              ┌───────────┴───────────┐                                     │
│              │  ACTION-DRIVEN VIDEO  │                                     │
│              │        PARSER         │                                     │
│              └───────────┬───────────┘                                     │
│                          │                                                  │
│    ┌─────────────────────┼─────────────────────┐                           │
│    │                     │                     │                            │
│    ▼                     ▼                     ▼                            │
│ ┌────────┐         ┌──────────┐         ┌───────────┐                      │
│ │Mask2For│         │DepthAny- │         │  Qwen3-VL │                      │
│ │mer+Swin│         │thingV3   │         │   (8B)    │                      │
│ │Segment │         │  Depth   │         │  Actions  │                      │
│ └───┬────┘         └────┬─────┘         └─────┬─────┘                      │
│     └───────────────────┼─────────────────────┘                            │
│                         │                                                   │
│              ┌──────────┴──────────┐                                       │
│              │ STREAMING SURGICAL  │                                       │
│              │       VIDEO         │                                       │
│              └─────────────────────┘                                       │
│                   VIDEO PROCESSING LAYER                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   OUTPUT: Progress &    │
                    │   Safety Checkpoints    │
                    │  "Step 3 Complete ✓"    │
                    │  "⚠ Step 5 Skipped"     │
                    └─────────────────────────┘
```

### 4.2 Pipeline Stage A: Action-Driven Video Parser

**Purpose**: Convert high-dimensional pixel streams into structured, text-serialized ActDTs

#### A1. Spatio-Temporal Segmentation
- **Model**: Mask2Former with VideoSwin backbone (pretrained on surgical videos)
- **Output**: Pixel-precise segmentation masks for 29+ surgical objects/anatomies
- **Temporal**: VideoSwin processes consecutive frame chunks for motion dynamics

#### A2. 3D Coordinate Extraction
- **Model**: DepthAnythingV3
- **Output**: Centroid depth for each segmented element
- **Format**: ℝ⁵ coordinates (x₁, y₁, x₂, y₂, z_depth)

#### A3. Action Attribute Generation
- **Model**: Qwen3-VL-8B (fine-tuned on surgical interactions)
- **Input**: Localized visual regions of interest
- **Output**: Natural language action attributes (tool-tissue interactions)

#### A4. JSON Serialization
```json
{
  "clip_id": "video_001_clip_042",
  "interval": [125.0, 130.0],
  "actions": [
    {
      "subject": "Capsulorhexis Forceps",
      "action": "Tear",
      "object": "Anterior Capsule",
      "location_3D": [0.32, 0.45, 0.38, 0.52, 0.67],
      "confidence": 0.94
    },
    {
      "subject": "Rycroft Cannula",
      "action": "Inject",
      "object": "Anterior Chamber",
      "location_3D": [0.55, 0.48, 0.61, 0.54, 0.45],
      "confidence": 0.89
    }
  ],
  "predicted_step": "Capsulorhexis",
  "predicted_phase": "Nucleus Removal Preparation"
}
```

### 4.3 Pipeline Stage B: Spatio-Temporal Hierarchical Aligner

#### B1. Hierarchical Relation Aggregation Module (HRAM)
- **Function**: Learn semantic relations across task hierarchies
- **Mechanism**: Cross-attention blocks linking Phase ↔ Step ↔ Action features
- **Benefit**: Fine-grained tool-action features assist high-level step localization

#### B2. Disentangled Multi-Space Alignment (HecVL-style)

| Space | Granularity | Text Type | Visual Aggregation |
|-------|-------------|-----------|-------------------|
| S_narration | Clip-level | Action triplets | Single clip features |
| S_concept | Step-level | Conceptual summaries | Aggregated clip sequences |
| S_abstract | Phase-level | Procedure abstracts | Full phase features |

#### B3. Inter-Task Contrastive Learning (ICL)

**Loss Function**:
$$\mathcal{L}_{ICL} = \sum_{(i,j) \in \mathcal{P}} \mathcal{L}_{InfoNCE}(v_i, t_j) + \lambda \cdot \mathcal{L}_{consistency}(h_i, h_j)$$

Where:
- $(i,j) \in \mathcal{P}$ = consistent task pairs (Phase↔Step, Action↔Instrument)
- $\mathcal{L}_{consistency}$ = regularization for hierarchical agreement
- $\lambda$ = weighting hyperparameter

### 4.4 Pipeline Stage C: Temporal Robustness Mechanisms

#### C1. Step-Aware Feature Representation (SFR) — From StepAL

**Problem**: Skipped steps create dimensional inconsistency

**Solution**:
1. Organize clip embeddings by predicted step pseudo-labels
2. If canonical step $S_k$ is omitted, pad with global video average:

$$\mathbf{f}_{S_k} = \begin{cases} \frac{1}{|C_k|}\sum_{c \in C_k} \mathbf{f}_c & \text{if } |C_k| > 0 \\ \frac{1}{|V|}\sum_{c \in V} \mathbf{f}_c & \text{if step skipped} \end{cases}$$

#### C2. Temporal Rectification — From SurgPLAN++

**Problem**: Online streaming lacks global context

**Solution**:
1. **Mirroring**: Extend partial video by reversing existing content
2. **Center-Duplication**: Replicate middle segments to simulate completion
3. **Dynamic Rectification**: Continuously refine predictions as new frames arrive

$$R_{t+1} = \text{Rectify}(R_t, P_{global}(V_{0:t+1}))$$

---

## 5. Handling Cross-Procedure Generalization

### 5.1 Open-Vocabulary Checklist Substitution

**Key Insight**: The neural network never changes; only the text input file changes.

| Surgery Type | Phase Example | Step Example | Instrument Example |
|--------------|---------------|--------------|-------------------|
| **Cataract** | Nucleus Removal | Capsulorhexis | Phaco Handpiece |
| **Cholecystectomy** | Calot Triangle Dissection | Clipping Cystic Artery | Clip Applier |
| **Prostatectomy** | Bladder Neck Transection | Posterior Dissection | Robotic Grasper |
| **Gastric Bypass** | Gastric Pouch Creation | Pouch Stapling | Linear Stapler |

### 5.2 Transfer Mechanism

```python
# Pseudo-code for cross-procedure transfer
def transfer_to_new_surgery(model, new_checklist_path):
    """
    No retraining required - just load new canonical texts
    """
    # Load new canonical descriptions
    new_canonical = load_json(new_checklist_path)
    
    # Encode new texts in same embedding spaces
    phase_embeddings = model.text_encoder(new_canonical['phases'], space='abstract')
    step_embeddings = model.text_encoder(new_canonical['steps'], space='concept')
    action_embeddings = model.text_encoder(new_canonical['actions'], space='narration')
    
    # Model ready for zero-shot inference
    return model.with_canonical(phase_embeddings, step_embeddings, action_embeddings)
```

### 5.3 Evidence-Grounded Refinement (Optional Enhancement)

For deployment at hospitals with significant protocol deviations:
1. LLM analyzes discrepancies between canonical expectations and observed patterns
2. Dynamically adjusts step definitions over iterative rounds
3. Adapts to center-specific clinical realities without full retraining

---

# PART III: EVALUATION FRAMEWORK

## 6. Three-Tier Evaluation Suite

### Tier 1: Zero-Shot Recognition Benchmarks

| Metric | Formula/Description | Target |
|--------|---------------------|--------|
| **Frame Accuracy** | Correct frame classifications / Total frames | >85% |
| **Step Accuracy** | Correct step predictions / Total steps | >75% |
| **Phase F1-Score** | Harmonic mean of precision and recall | >80% |
| **Video Jaccard (IoU)** | Temporal overlap of predicted vs. GT segments | >70% |
| **Recall@K** | % correct retrievals in top K results | R@1 >40%, R@5 >70% |

### Tier 2: Alignment Quality & Process Integrity

| Metric | Formula | Clinical Relevance |
|--------|---------|-------------------|
| **Alignment Discrepancy Index (ADI)** | $\frac{1}{M}\sum_{i=1}^{M}(\|t^{GT}_{start,i} - t^{Pred}_{start,i}\| + \|t^{GT}_{end,i} - t^{Pred}_{end,i}\|)$ | Temporal precision of alignment |
| **Sequence Order Integrity (SOI)** | Kendall's τ correlation with canonical order | Chronological correctness |
| **Hierarchical Consistency Score (HCS)** | % frames where Phase-Step-Action agree | Multi-level coherence |
| **Checkpoint Detection Recall (CDR)** | Correctly flagged milestones / Total milestones | Safety verification |

### Tier 3: Anomaly & Cross-Center Robustness

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Skip Detection Recall (SDR)** | Correctly flagged skips / Actual skips | Safety alerting |
| **False Alarm Rate (FAR)** | Incorrect skip flags / Total flags | Avoiding alert fatigue |
| **Cross-Center Drop Rate** | (Acc_trained_center - Acc_new_center) / Acc_trained_center | Generalization robustness |
| **Zero-Shot Transfer Accuracy** | Performance on completely unseen surgery types | Deployment flexibility |

### Novel Composite Metric: Procedural Alignment Score (PAS)

$$\text{PAS} = 0.3 \times \text{Phase\_Acc} + 0.4 \times \text{Step\_Acc} + 0.3 \times \text{SDR}$$

This captures both recognition accuracy and safety-critical anomaly detection.

---

## 7. Experimental Design

### 7.1 Experiment Matrix

| Exp ID | Experiment | Datasets | Primary Metrics |
|--------|------------|----------|-----------------|
| E1 | Single-surgery supervised baseline | CATARACTS | Phase Acc, Step Acc |
| E2 | Zero-shot transfer: Cataract → Cholecystectomy | CAT-SG → Cholec80 | Zero-Shot Acc, PAS |
| E3 | Zero-shot transfer: Cataract → Prostatectomy | CAT-SG → PSI-AVA | Zero-Shot Acc, PAS |
| E4 | Cross-center robustness | StrasBypass70 ↔ BernBypass70 | Drop Rate, SOI |
| E5 | Ablation: Single vs. hierarchical embedding | CAT-SG | Phase Acc, HCS |
| E6 | Ablation: With/without ICL loss | CAT-SG | Step Acc, HCS |
| E7 | Ablation: With/without SFR padding | Cataract-1k | SDR, FAR |
| E8 | Ablation: With/without temporal rectification | StrasBypass70 | SOI, ADI |
| E9 | Skip detection stress test | Synthetic + Real | SDR, FAR, PAS |
| E10 | Real-time latency benchmark | All | ms/frame, FPS |

### 7.2 Baseline Comparisons

| Method | Type | What It Represents |
|--------|------|-------------------|
| Trans-SVNet | Supervised | Current SOTA surgical phase recognition |
| CLIP4Clip | Zero-shot | Standard video-language retrieval |
| HecVL (reproduced) | Zero-shot | Hierarchical VL without ActDT |
| Ours (USA) | Zero-shot | Full proposed framework |

---

# PART IV: DATA REQUIREMENTS

## 8. Dataset Inventory

### 8.1 Primary Datasets (Required)

| Dataset | Surgery | Annotations | Size | Role |
|---------|---------|-------------|------|------|
| **CAT-SG / CaDIS** | Cataract | 29 classes, 1.8M relations, scene graphs | 56 videos, 164K frames | Primary benchmark; ActDT validation |
| **CATARACTS** | Cataract | 19 steps, tool annotations | 50 videos | Step recognition baseline |
| **Cataract-1k** | Cataract | 13 chronological steps | 1000 videos | SFR evaluation; scale testing |
| **Cholec80** | Cholecystectomy | 7 phases, tool presence | 80 videos | Cross-procedure transfer target |
| **PSI-AVA** | Robotic Prostatectomy | 11 phases, 21 steps, 16 actions, 7 instruments | Multi-center | Cross-specialty transfer |
| **StrasBypass70 / BernBypass70** | Gastric Bypass | Phase/step labels (two centers) | 140 videos total | Multi-center robustness |

### 8.2 Pretraining Datasets

| Dataset | Content | Size | Purpose |
|---------|---------|------|---------|
| **SurgVLP / SVL-25K** | Multi-procedure lecture videos | 25K clips, 10K concepts, 1K abstracts | Visual-textual encoder pretraining |
| **MedVideoCap-55K** | Diverse clinical/imaging clips | 55K videos with captions | Large-scale generalization |
| **HowTo100M** (surgical subset) | Instructional videos | Filtered subset | General video-language foundations |

### 8.3 Artifacts to Create

| Artifact | Description | Effort | Priority |
|----------|-------------|--------|----------|
| Canonical JSON templates | MSSU hierarchy for each surgery type | Medium | HIGH |
| Cross-procedure vocabulary | Standardized instrument/action ontology | High | HIGH |
| Evaluation splits | Zero-shot vs. fine-tuned partitions | Low | MEDIUM |
| Synthetic skip sequences | Artificially created skip scenarios | Medium | MEDIUM |

---

# PART V: IMPLEMENTATION ROADMAP

## 9. Master Timeline (12 Months)

```
Month 1-2     Month 3-4     Month 5-6     Month 7-8     Month 9-10    Month 11-12
────────────────────────────────────────────────────────────────────────────────
[FOUNDATION]  [PARSER DEV]  [ALIGNER DEV] [ROBUSTNESS]  [EVALUATION]  [WRITING]
                                                                      
Data Prep     Segmentation  HCT+ & HecVL  SFR Padding   Benchmarks    Paper Draft
Environment   Depth Extract Multi-Spaces  Rectification Multi-Center  Review Cycle
SurgVLP Prep  ActDT JSON    ICL Training  Skip Handling Ablations     Submission
```

---

## 10. Detailed Task Breakdown

### Phase 1: Foundation (Months 1-2)

#### Development Tasks

| ID | Task | Priority | Hours | Dependencies |
|----|------|----------|-------|--------------|
| D1.1 | Set up development environment (PyTorch 2.x, CUDA 12.x, W&B) | HIGH | 4 | None |
| D1.2 | Configure multi-GPU training infrastructure | HIGH | 8 | D1.1 |
| D1.3 | Download and verify CATARACTS dataset | HIGH | 4 | D1.1 |
| D1.4 | Download and verify CAT-SG dataset | HIGH | 4 | D1.1 |
| D1.5 | Download and verify Cholec80 dataset | HIGH | 4 | D1.1 |
| D1.6 | Download and verify PSI-AVA dataset | MEDIUM | 6 | D1.1 |
| D1.7 | Download and verify StrasBypass70/BernBypass70 | MEDIUM | 6 | D1.1 |
| D1.8 | Preprocess all videos to 1fps standardized format | HIGH | 12 | D1.3-D1.7 |
| D1.9 | Download pretrained backbones (VideoSwin, BioClinicalBERT) | HIGH | 4 | D1.1 |
| D1.10 | Download Qwen3-VL-8B checkpoint | HIGH | 4 | D1.1 |
| D1.11 | Set up SurgVLP pretraining data pipeline | MEDIUM | 16 | D1.1 |
| D1.12 | Create unified data loading framework | HIGH | 16 | D1.8 |
| D1.13 | Implement basic evaluation harness | MEDIUM | 12 | D1.1 |
| D1.14 | Set up experiment tracking (W&B dashboards) | LOW | 4 | D1.1 |

**Phase 1 Development Total: ~104 hours**

#### Writing Tasks

| ID | Task | Priority | Hours | Dependencies |
|----|------|----------|-------|--------------|
| W1.1 | Create detailed literature review outline | HIGH | 6 | None |
| W1.2 | Deep-dive notes: HecVL paper | HIGH | 8 | None |
| W1.3 | Deep-dive notes: HCT (MSSU) paper | HIGH | 8 | None |
| W1.4 | Deep-dive notes: OR3 / ActDT paper | HIGH | 8 | None |
| W1.5 | Deep-dive notes: StepAL paper | HIGH | 6 | None |
| W1.6 | Deep-dive notes: SurgPLAN++ paper | HIGH | 6 | None |
| W1.7 | Deep-dive notes: Dynamic Scene Graphs paper | MEDIUM | 4 | None |
| W1.8 | Draft Related Work section v1 | MEDIUM | 12 | W1.2-W1.7 |
| W1.9 | Create problem statement diagram | MEDIUM | 4 | W1.8 |

**Phase 1 Writing Total: ~62 hours**

---

### Phase 2: Action-Driven Video Parser (Months 3-4)

#### Development Tasks

| ID | Task | Priority | Hours | Dependencies |
|----|------|----------|-------|--------------|
| D2.1 | Implement Mask2Former segmentation module | HIGH | 20 | D1.9 |
| D2.2 | Fine-tune Mask2Former on CaDIS surgical classes | HIGH | 24 | D2.1 |
| D2.3 | Integrate VideoSwin backbone for temporal features | HIGH | 16 | D2.1 |
| D2.4 | Implement DepthAnythingV3 integration | HIGH | 12 | D1.1 |
| D2.5 | Build centroid extraction from segmentation masks | MEDIUM | 8 | D2.1, D2.4 |
| D2.6 | Combine segmentation + depth into ℝ⁵ coordinates | HIGH | 8 | D2.5 |
| D2.7 | Implement Qwen3-VL action attribute generator | HIGH | 16 | D1.10 |
| D2.8 | Create ROI extraction for VLM input | MEDIUM | 8 | D2.1, D2.7 |
| D2.9 | Design ActDT JSON schema | HIGH | 4 | None |
| D2.10 | Implement JSON serialization pipeline | HIGH | 8 | D2.9 |
| D2.11 | Build end-to-end video-to-ActDT pipeline | HIGH | 16 | D2.1-D2.10 |
| D2.12 | Optimize pipeline for batch processing | MEDIUM | 12 | D2.11 |
| D2.13 | Unit tests for each parser component | MEDIUM | 8 | D2.11 |
| D2.14 | Generate ActDTs for all training videos | HIGH | 20 | D2.11 |

**Phase 2 Development Total: ~180 hours**

#### Writing Tasks

| ID | Task | Priority | Hours | Dependencies |
|----|------|----------|-------|--------------|
| W2.1 | Draft "Video Parser Architecture" methodology section | HIGH | 10 | D2.11 |
| W2.2 | Create parser pipeline diagram | HIGH | 4 | D2.11 |
| W2.3 | Document ActDT JSON schema with examples | MEDIUM | 4 | D2.10 |
| W2.4 | Write segmentation evaluation results | MEDIUM | 4 | D2.14 |

**Phase 2 Writing Total: ~22 hours**

---

### Phase 3: Canonical Reference & Hierarchical Aligner (Months 5-6)

#### Development Tasks

| ID | Task | Priority | Hours | Dependencies |
|----|------|----------|-------|--------------|
| D3.1 | Define MSSU hierarchy for cataract surgery (JSON) | HIGH | 8 | None |
| D3.2 | Define MSSU hierarchy for cholecystectomy (JSON) | HIGH | 8 | None |
| D3.3 | Define MSSU hierarchy for prostatectomy (JSON) | MEDIUM | 8 | None |
| D3.4 | Define MSSU hierarchy for gastric bypass (JSON) | MEDIUM | 8 | None |
| D3.5 | Create cross-procedure instrument ontology | HIGH | 16 | D3.1-D3.4 |
| D3.6 | Implement BioClinicalBERT text encoder | HIGH | 12 | D1.9 |
| D3.7 | Create three separate projection heads (abstract/concept/narration) | HIGH | 12 | D3.6 |
| D3.8 | Implement temporal aggregation module (clip → step → phase) | HIGH | 16 | D3.7 |
| D3.9 | Implement HRAM cross-attention module | HIGH | 20 | D3.8 |
| D3.10 | Implement InfoNCE contrastive loss | HIGH | 8 | None |
| D3.11 | Implement Inter-task Contrastive Learning (ICL) loss | HIGH | 12 | D3.10 |
| D3.12 | Build multi-granularity training loop | HIGH | 16 | D3.7-D3.11 |
| D3.13 | Implement curriculum learning schedule | MEDIUM | 8 | D3.12 |
| D3.14 | Train visual-textual encoders on SurgVLP | HIGH | 40 | D3.12, D1.11 |
| D3.15 | Fine-tune on CAT-SG with canonical alignment | HIGH | 32 | D3.14 |
| D3.16 | Hyperparameter tuning (LR, batch size, temperature) | MEDIUM | 24 | D3.15 |

**Phase 3 Development Total: ~248 hours**

#### Writing Tasks

| ID | Task | Priority | Hours | Dependencies |
|----|------|----------|-------|--------------|
| W3.1 | Draft "Canonical Reference Design" section | HIGH | 8 | D3.1-D3.5 |
| W3.2 | Draft "Hierarchical Aligner" methodology section | HIGH | 12 | D3.15 |
| W3.3 | Document loss function formulations (LaTeX) | HIGH | 6 | D3.11 |
| W3.4 | Create HRAM architecture diagram | MEDIUM | 4 | D3.9 |
| W3.5 | Create multi-space alignment visualization | MEDIUM | 4 | D3.7 |

**Phase 3 Writing Total: ~34 hours**

---

### Phase 4: Temporal Robustness Mechanisms (Months 7-8)

#### Development Tasks

| ID | Task | Priority | Hours | Dependencies |
|----|------|----------|-------|--------------|
| D4.1 | Implement step pseudo-labeling for SFR | HIGH | 12 | D3.15 |
| D4.2 | Implement SFR feature organization by predicted step | HIGH | 12 | D4.1 |
| D4.3 | Implement global average padding for skipped steps | HIGH | 8 | D4.2 |
| D4.4 | Build SFR-enhanced feature representation module | HIGH | 12 | D4.3 |
| D4.5 | Implement video mirroring augmentation | HIGH | 8 | None |
| D4.6 | Implement center-duplication augmentation | HIGH | 8 | None |
| D4.7 | Implement down-sampling for pseudo-complete sequences | MEDIUM | 6 | None |
| D4.8 | Build SurgPLAN++ temporal rectification loop | HIGH | 20 | D4.5-D4.7 |
| D4.9 | Implement dynamic results sequence updater | HIGH | 12 | D4.8 |
| D4.10 | Create synthetic skip sequences for testing | MEDIUM | 12 | D1.8 |
| D4.11 | Implement skip detection alerting logic | HIGH | 8 | D4.4 |
| D4.12 | Implement checkpoint verification module | MEDIUM | 8 | D4.11 |
| D4.13 | Integrate all robustness modules into main pipeline | HIGH | 16 | D4.4, D4.9 |
| D4.14 | End-to-end robustness testing | HIGH | 16 | D4.13 |

**Phase 4 Development Total: ~148 hours**

#### Writing Tasks

| ID | Task | Priority | Hours | Dependencies |
|----|------|----------|-------|--------------|
| W4.1 | Draft "Handling Skipped Steps" methodology section | HIGH | 8 | D4.13 |
| W4.2 | Draft "Temporal Rectification" methodology section | HIGH | 8 | D4.9 |
| W4.3 | Create SFR padding visualization | MEDIUM | 4 | D4.4 |
| W4.4 | Create rectification process diagram | MEDIUM | 4 | D4.9 |

**Phase 4 Writing Total: ~24 hours**

---

### Phase 5: Evaluation & Experiments (Months 9-10)

#### Development Tasks

| ID | Task | Priority | Hours | Dependencies |
|----|------|----------|-------|--------------|
| D5.1 | Implement all Tier 1 metrics (Acc, F1, Jaccard, R@K) | HIGH | 12 | None |
| D5.2 | Implement Tier 2 metrics (ADI, SOI, HCS, CDR) | HIGH | 16 | None |
| D5.3 | Implement Tier 3 metrics (SDR, FAR, Drop Rate) | HIGH | 12 | None |
| D5.4 | Implement PAS composite metric | MEDIUM | 4 | D5.1-D5.3 |
| D5.5 | Implement zero-shot evaluation protocol | HIGH | 12 | D5.1 |
| D5.6 | Run E1: Supervised baseline on CATARACTS | HIGH | 8 | D3.15 |
| D5.7 | Run E2: Zero-shot Cataract → Cholecystectomy | HIGH | 12 | D5.5 |
| D5.8 | Run E3: Zero-shot Cataract → Prostatectomy | HIGH | 12 | D5.5 |
| D5.9 | Run E4: Cross-center (Strasbourg ↔ Bern) | HIGH | 16 | D5.5 |
| D5.10 | Run E5: Ablation - single vs. hierarchical | HIGH | 8 | D5.5 |
| D5.11 | Run E6: Ablation - with/without ICL | HIGH | 8 | D5.5 |
| D5.12 | Run E7: Ablation - with/without SFR | HIGH | 8 | D5.5 |
| D5.13 | Run E8: Ablation - with/without rectification | HIGH | 8 | D5.5 |
| D5.14 | Run E9: Skip detection stress test | HIGH | 12 | D4.10 |
| D5.15 | Run E10: Real-time latency benchmark | MEDIUM | 8 | D4.13 |
| D5.16 | Statistical significance testing (paired t-tests) | MEDIUM | 8 | D5.6-D5.15 |
| D5.17 | Error analysis and failure case extraction | MEDIUM | 12 | D5.6-D5.15 |
| D5.18 | Generate qualitative visualizations | MEDIUM | 12 | D5.17 |

**Phase 5 Development Total: ~178 hours**

#### Writing Tasks

| ID | Task | Priority | Hours | Dependencies |
|----|------|----------|-------|--------------|
| W5.1 | Draft "Experimental Setup" section | HIGH | 10 | D5.5 |
| W5.2 | Create main results tables | HIGH | 8 | D5.6-D5.9 |
| W5.3 | Create ablation study tables | HIGH | 6 | D5.10-D5.13 |
| W5.4 | Create cross-center comparison figures | MEDIUM | 4 | D5.9 |
| W5.5 | Write "Results" section with analysis | HIGH | 16 | W5.2-W5.4 |
| W5.6 | Write "Discussion" section | HIGH | 12 | W5.5 |
| W5.7 | Create qualitative visualization figures | MEDIUM | 8 | D5.18 |
| W5.8 | Write error analysis subsection | MEDIUM | 6 | D5.17 |

**Phase 5 Writing Total: ~70 hours**

---

### Phase 6: Paper Writing & Submission (Months 11-12)

#### Writing Tasks

| ID | Task | Priority | Hours | Dependencies |
|----|------|----------|-------|--------------|
| W6.1 | Write Abstract (250 words) | HIGH | 4 | W5.5 |
| W6.2 | Write Introduction section | HIGH | 16 | W1.8 |
| W6.3 | Finalize Related Work section | HIGH | 10 | W1.8 |
| W6.4 | Finalize complete Methodology section | HIGH | 12 | W2.1, W3.2, W4.1 |
| W6.5 | Finalize Experiments section | HIGH | 10 | W5.1-W5.8 |
| W6.6 | Write Conclusion section | HIGH | 6 | W5.6 |
| W6.7 | Write Limitations and Future Work | MEDIUM | 6 | W5.6 |
| W6.8 | Format all references (BibTeX cleanup) | MEDIUM | 6 | All |
| W6.9 | Create all camera-ready figures | HIGH | 16 | All |
| W6.10 | Create supplementary materials | MEDIUM | 12 | All |
| W6.11 | Internal review and revision cycle 1 | HIGH | 20 | W6.1-W6.10 |
| W6.12 | Advisor review incorporation | HIGH | 16 | W6.11 |
| W6.13 | Revision cycle 2 | HIGH | 12 | W6.12 |
| W6.14 | Final proofreading | HIGH | 8 | W6.13 |
| W6.15 | Format for target venue | MEDIUM | 4 | W6.14 |
| W6.16 | Submit to venue | HIGH | 2 | W6.15 |

**Phase 6 Writing Total: ~160 hours**

---

## 11. Summary Statistics

### Total Effort by Phase

| Phase | Development | Writing | Total |
|-------|-------------|---------|-------|
| Phase 1: Foundation | 104 hrs | 62 hrs | 166 hrs |
| Phase 2: Video Parser | 180 hrs | 22 hrs | 202 hrs |
| Phase 3: Hierarchical Aligner | 248 hrs | 34 hrs | 282 hrs |
| Phase 4: Robustness | 148 hrs | 24 hrs | 172 hrs |
| Phase 5: Evaluation | 178 hrs | 70 hrs | 248 hrs |
| Phase 6: Writing | — | 160 hrs | 160 hrs |
| **TOTAL** | **858 hrs** | **372 hrs** | **1,230 hrs** |

### Weekly Commitment
- **12 months × 4 weeks = 48 weeks**
- **1,230 hours / 48 weeks ≈ 25.6 hours/week**

---

## 12. Critical Path Items

### 🔴 Highest Priority Development (Blocks Everything)

| Rank | Task ID | Task | Why Critical |
|------|---------|------|--------------|
| 1 | D2.11 | End-to-end video-to-ActDT pipeline | Core technical contribution; blocks all alignment work |
| 2 | D3.5 | Cross-procedure instrument ontology | Blocks all cross-surgery experiments |
| 3 | D3.11 | ICL loss implementation | Key to hierarchical consistency claims |
| 4 | D4.4 | SFR-enhanced feature module | Key to skip-handling claims |
| 5 | D5.7-D5.9 | Zero-shot transfer experiments | Main paper claims validated here |

### 🔴 Highest Priority Writing (Blocks Submission)

| Rank | Task ID | Task | Why Critical |
|------|---------|------|--------------|
| 1 | W1.2-W1.6 | Paper deep-dives | Required before methodology design |
| 2 | W3.3 | Loss function documentation | Reviewer scrutiny area |
| 3 | W5.5 | Results analysis | Makes or breaks the paper |
| 4 | W6.2 | Introduction | Sets entire paper narrative |
| 5 | W6.12 | Advisor review | Catches major issues before submission |

---

## 13. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dataset access delays | Medium | High | Start download requests immediately; identify backup datasets |
| GPU resource constraints | Medium | High | Reserve cloud GPU credits; optimize batch sizes early |
| Qwen3-VL integration issues | Medium | Medium | Have BLIP-2 as backup VLM |
| Cross-center data unavailable | Low | High | Use synthetic center variations; focus on cross-procedure |
| Performance below baselines | Low | High | Extensive ablations to identify failure modes; adjust architecture |

---

## 14. Target Venues

| Venue | Deadline | Fit Score | Notes |
|-------|----------|-----------|-------|
| **MICCAI 2025** | ~March 2025 | ⭐⭐⭐⭐⭐ | Primary target; top medical imaging venue |
| **IPCAI 2025** | ~December 2024 | ⭐⭐⭐⭐ | Surgical AI focus; good backup |
| **CVPR 2025** | ~November 2024 | ⭐⭐⭐ | Broader audience; higher competition |
| **Medical Image Analysis** | Rolling | ⭐⭐⭐⭐⭐ | Journal for extended version |
| **IEEE TMI** | Rolling | ⭐⭐⭐⭐ | Strong medical imaging journal |

---

## 15. Immediate Next Actions

### This Week
- [ ] Set up Git repository with project structure
- [ ] Begin HecVL paper deep-read (W1.2)
- [ ] Submit dataset access requests (CATARACTS, CAT-SG, PSI-AVA)
- [ ] Configure development environment (D1.1)

### Next Week
- [ ] Complete HecVL notes
- [ ] Begin OR3 paper deep-read (W1.4)
- [ ] Download and verify first dataset
- [ ] Draft MSSU hierarchy for cataract surgery

### Month 1 Milestone
- [ ] All paper deep-dives complete
- [ ] Development environment fully configured
- [ ] At least 3 datasets downloaded and verified
- [ ] Related Work section v1 drafted

---

# APPENDIX A: Canonical Hierarchy Examples

## Cataract Surgery

```json
{
  "procedure": "Cataract Surgery",
  "phases": [
    {
      "id": "P1",
      "name": "Preparation",
      "steps": [
        {"id": "S1", "name": "Sterile Draping", "actions": ["Drape", "Position"]},
        {"id": "S2", "name": "Anesthesia", "actions": ["Inject", "Apply"]}
      ]
    },
    {
      "id": "P2", 
      "name": "Incision",
      "steps": [
        {"id": "S3", "name": "Main Incision", "actions": ["Cut"], "instruments": ["Primary Knife"], "anatomy": ["Cornea"]},
        {"id": "S4", "name": "Side Port", "actions": ["Cut"], "instruments": ["Secondary Knife"], "anatomy": ["Cornea"]}
      ]
    },
    {
      "id": "P3",
      "name": "Capsulorhexis",
      "steps": [
        {"id": "S5", "name": "Viscoelastic Injection", "actions": ["Inject"], "instruments": ["Rycroft Cannula"], "anatomy": ["Anterior Chamber"]},
        {"id": "S6", "name": "Capsule Tearing", "actions": ["Tear", "Grasp"], "instruments": ["Capsulorhexis Forceps"], "anatomy": ["Anterior Capsule"]}
      ]
    }
  ]
}
```

## Cholecystectomy

```json
{
  "procedure": "Laparoscopic Cholecystectomy",
  "phases": [
    {
      "id": "P1",
      "name": "Preparation",
      "steps": [
        {"id": "S1", "name": "Trocar Placement", "actions": ["Insert", "Inflate"]}
      ]
    },
    {
      "id": "P2",
      "name": "Calot Triangle Dissection",
      "steps": [
        {"id": "S2", "name": "Cystic Duct Identification", "actions": ["Dissect", "Expose"], "instruments": ["Grasper", "Hook"], "anatomy": ["Cystic Duct"]},
        {"id": "S3", "name": "Cystic Artery Identification", "actions": ["Dissect", "Expose"], "instruments": ["Grasper", "Hook"], "anatomy": ["Cystic Artery"]}
      ]
    },
    {
      "id": "P3",
      "name": "Clipping and Cutting",
      "steps": [
        {"id": "S4", "name": "Clip Cystic Duct", "actions": ["Clip"], "instruments": ["Clip Applier"], "anatomy": ["Cystic Duct"]},
        {"id": "S5", "name": "Clip Cystic Artery", "actions": ["Clip"], "instruments": ["Clip Applier"], "anatomy": ["Cystic Artery"]},
        {"id": "S6", "name": "Cut Cystic Duct", "actions": ["Cut"], "instruments": ["Scissors"], "anatomy": ["Cystic Duct"]}
      ]
    }
  ]
}
```

---

# APPENDIX B: Evaluation Metric Implementations

```python
import numpy as np
from scipy.stats import kendalltau

def alignment_discrepancy_index(gt_boundaries, pred_boundaries):
    """
    ADI: Average temporal offset between GT and predicted step boundaries
    
    Args:
        gt_boundaries: List of (start, end) tuples for ground truth
        pred_boundaries: List of (start, end) tuples for predictions
    
    Returns:
        ADI score (lower is better)
    """
    assert len(gt_boundaries) == len(pred_boundaries)
    
    total_discrepancy = 0
    for (gt_start, gt_end), (pred_start, pred_end) in zip(gt_boundaries, pred_boundaries):
        total_discrepancy += abs(gt_start - pred_start) + abs(gt_end - pred_end)
    
    return total_discrepancy / len(gt_boundaries)


def sequence_order_integrity(canonical_order, predicted_order):
    """
    SOI: Kendall's tau correlation between canonical and predicted step orders
    
    Args:
        canonical_order: List of step IDs in canonical order
        predicted_order: List of step IDs in predicted order
    
    Returns:
        Kendall's tau coefficient (-1 to 1, higher is better)
    """
    # Convert to ranks
    canonical_ranks = {step: i for i, step in enumerate(canonical_order)}
    pred_ranks = [canonical_ranks.get(step, len(canonical_order)) for step in predicted_order]
    
    tau, _ = kendalltau(list(range(len(predicted_order))), pred_ranks)
    return tau


def hierarchical_consistency_score(phase_preds, step_preds, action_preds, hierarchy_map):
    """
    HCS: Percentage of frames where Phase-Step-Action predictions are consistent
    
    Args:
        phase_preds: Array of phase predictions per frame
        step_preds: Array of step predictions per frame
        action_preds: Array of action predictions per frame
        hierarchy_map: Dict mapping steps to valid phases, actions to valid steps
    
    Returns:
        HCS percentage (0-100, higher is better)
    """
    consistent_frames = 0
    total_frames = len(phase_preds)
    
    for phase, step, action in zip(phase_preds, step_preds, action_preds):
        phase_step_consistent = step in hierarchy_map['phase_to_steps'].get(phase, [])
        step_action_consistent = action in hierarchy_map['step_to_actions'].get(step, [])
        
        if phase_step_consistent and step_action_consistent:
            consistent_frames += 1
    
    return (consistent_frames / total_frames) * 100


def skip_detection_metrics(gt_skips, pred_skips, total_flags):
    """
    SDR and FAR: Skip detection recall and false alarm rate
    
    Args:
        gt_skips: Set of actually skipped step IDs
        pred_skips: Set of predicted skipped step IDs
        total_flags: Total number of skip flags raised
    
    Returns:
        (SDR, FAR) tuple
    """
    true_positives = len(gt_skips & pred_skips)
    false_positives = len(pred_skips - gt_skips)
    
    sdr = true_positives / len(gt_skips) if gt_skips else 1.0
    far = false_positives / total_flags if total_flags else 0.0
    
    return sdr, far


def procedural_alignment_score(phase_acc, step_acc, sdr):
    """
    PAS: Composite metric for overall alignment quality
    
    Args:
        phase_acc: Phase accuracy (0-1)
        step_acc: Step accuracy (0-1)
        sdr: Skip detection recall (0-1)
    
    Returns:
        PAS score (0-1, higher is better)
    """
    return 0.3 * phase_acc + 0.4 * step_acc + 0.3 * sdr
```

---

# APPENDIX C: Key Paper Reference Summary

| Paper | Key Technique | What to Extract | Gap It Leaves |
|-------|---------------|-----------------|---------------|
| **HecVL** | Hierarchical video-language pretraining | Multi-space contrastive learning architecture | No forward parser; no ActDT; no skip handling |
| **HCT (MSSU)** | Multi-level semantic understanding | HRAM cross-attention; ICL loss formulation | Fully supervised; no zero-shot |
| **OR3** | Action-driven digital twins | ActDT JSON schema; reasoning retrieval | Inverse direction only |
| **StepAL** | Step-aware feature representation | SFR padding mechanism | No cross-modal alignment |
| **SurgPLAN++** | Temporal phase localization | Mirroring/duplication; rectification loop | Vision-only; no language |
| **CAT-SG** | Dynamic scene graphs | Graph structure; 29-class ontology | No video-language alignment |

---

*This master proposal synthesizes the technical depth of the AI-refined version with the comprehensive task breakdown and practical execution focus. It provides a complete roadmap from literature review through submission, with clear milestones, metrics, and contingencies.*