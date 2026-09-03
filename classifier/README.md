# Video Datasets for Hard Negative Mining

Here's a categorized list of **publicly available datasets** you can pull from, organized by which class they'd serve as negatives for.

## What this data feeds

All three approaches in this repo (`high_compute_nn_cascade/`,
`low_compute_nn_flat/`, `stacked_trees_cascade/`) consume the **same**
two inputs, built from the datasets below:

1. `raw_videos/` — a folder of `.mp4`/`.mov`/`.avi`/`.mkv` files (the
   only formats `extract_frames.py` currently reads).
2. `labels.csv` — one row per video, columns `video_id, is_surgery, surgery_type`:
   - `video_id` must exactly match the video's filename stem (e.g. a
     file `cataract_0042.mp4` needs `video_id = cataract_0042`) — this
     is how frames, embeddings, and labels get joined back together.
   - `is_surgery`: `1` for anything in the "Class 2" and "Class 3"
     tables below, `0` for anything in "Class 1".
   - `surgery_type`: `cataract` for Class 3 rows, `other_surgery` for
     Class 2 rows, blank for Class 1 rows (`is_surgery=0`).

So this dataset list isn't just background reading — every dataset you
pull from here needs to become a set of video files in `raw_videos/`
plus matching rows in `labels.csv` using exactly those three columns.

## Making this data actually train the model well

A few things worth doing deliberately, not just "grab everything above":

- **Cap the easy-negative flood.** Kinetics/Something-Something/ActivityNet
  are 20K–700K clips; your cataract-positive data (Class 3) tops out
  around 100–150 videos total across CATARACTS + Cataract-101. If you
  dump all of Kinetics in, Stage 1 (`is_surgery`) will learn "surgery
  footage looks nothing like the 99% of the data" and won't have seen
  enough hard negatives. Subsample the easy-negative datasets down to
  roughly the same order of magnitude as your surgery-side data (or at
  least cap it — a few hundred to low thousands of clips is plenty),
  and lean on the "Tip" callout below for the classes that actually
  look surgery-adjacent.
- **Stage 2 balance matters more than Stage 1.** `is_cataract` is
  trained only on surgery videos, and cataract vs. other-surgery access
  is the whole point of the "hard negative" framing here. Track the
  cataract : other-surgery ratio directly — `train_*_cascade.py`
  already prints this count before training Stage 2, so use it as a
  checkpoint before committing to a full run.
- **No video should appear in more than one split.** The training
  scripts split at the video level (one embedding row per video), so as
  long as `video_id` is unique per source video this is already safe —
  just don't split a single long surgical video into multiple clips
  with different `video_id`s that could land on both sides of a
  train/val split.
- **Match frame-extraction settings across sources.** Per the practical
  note below, keep `--n_frames` and resolution consistent across every
  dataset you pull in — CLIP shouldn't be able to tell "this is
  Kinetics vs. this is CATARACTS" from compression artifacts or frame
  rate alone.
- **Images-only datasets (DeepDR/EyePACS, CHAOS) aren't usable as-is.**
  `extract_frames.py` only reads video files. Either find a video
  subset of these, skip them, or convert stills into single-frame
  "videos" only if you're deliberately testing the image-domain-shift
  case — don't silently mix formats.

---

## Class 1 Negatives: "Not Surgery"

### Medical/Clinical (Hard Negatives — Same Domain)
| Dataset | Content | Notes |
|---|---|---|
| **Kvasir-Capsule** | Capsule endoscopy (diagnostic, no instruments) | Good "medical but not surgery" negative |
| **DeepDR / EyePACS** (images mostly) | Fundus photography, diabetic retinopathy screening | For eye-exam-not-surgery negatives (mostly stills — check for video subsets) |
| **CHAOS / medical imaging challenge datasets** | CT/MRI — different modality, useful as "definitely not surgery" | Often not video, use cautiously |

### General Action Recognition (Easy but Necessary Negatives)
| Dataset | Content | Size |
|---|---|---|
| **Kinetics-400/600/700** | Broad human actions (cooking, crafting, sports) | 400K–700K clips |
| **Something-Something v2** | Fine-grained hand-object interactions (great for "hands + tools, not surgery") | ~220K clips |
| **UCF101 / HMDB51** | Classic action recognition | Smaller, easy negatives |
| **ActivityNet** | Diverse human activities | 20K videos |
| **AVA (Atomic Visual Actions)** | Fine-grained temporal action labels | ~430 videos |

**Tip**: From Kinetics/Something-Something, specifically pull classes like *"sewing," "carving wood," "cutting meat," "using a microscope," "applying makeup to eye area"* — these mimic surgical hand-motion and close-up framing.

---

## Class 2 Negatives: "Surgery, Not Cataract"

### Laparoscopic/General Surgery
| Dataset | Content | Size | Access |
|---|---|---|---|
| **Cholec80** | Laparoscopic cholecystectomy | 80 videos | Free (request) |
| **HeiChole** | Laparoscopic cholecystectomy + phase/action labels | 33 videos | Free (request) |
| **m2cai16-tool/workflow** | Laparoscopic surgery, tool presence | ~15 videos | Free |
| **AutoLaparo** | Laparoscopic hysterectomy | 21 videos | Free (request) |
| **GLENDA** | Gynecologic laparoscopy (endometriosis) | ~400 clips | Free |
| **LapGyn4** | Gynecologic laparoscopy (4 sub-tasks) | Varies | Free (request) |
| **Bypass170 / StrasBypass70** | Gastric bypass surgery | 170 / 70 videos | Free (request) |
| **DresdenSurgicalAnatomy** | Laparoscopic anatomy recognition | 32 videos | Free |

### Robotic Surgery
| Dataset | Content | Size | Access |
|---|---|---|---|
| **JIGSAWS** | Robotic suturing, knot-tying, needle-passing (da Vinci) | ~200 trials | Free |
| **SAR-RARP50** | Robotic-assisted radical prostatectomy | 50 videos | Free |
| **PETRAW** | Robotic peg transfer training | 150 sequences | Free |
| **PSI-AVA** | Robotic prostatectomy, action/phase labels | 8 videos (long) | Free (request) |

### Endoscopy (Diagnostic/Surgical Boundary Cases)
| Dataset | Content | Size | Access |
|---|---|---|---|
| **Kvasir / HyperKvasir** | GI endoscopy (mostly diagnostic — good for "not surgery" too) | Large image/video set | Free |
| **EndoVis sub-challenges** (various years) | Instrument segmentation, various surgery types | Varies by year | Free (request) |

---

## Class 3 Diversity (Positive Class — Cataract Variety)
| Dataset | Content | Size |
|---|---|---|
| **CATARACTS** | Cataract surgery, tool detection | 50 videos |
| **Cataract-101** | Cataract surgery, phase labels | 101 videos |
| **CaDIS** | Cataract surgery, semantic segmentation | Derived from CATARACTS |

*(Not negatives, but listed for completeness — combine with the above for balanced sampling.)*

---

## Practical Notes

1. **Access barriers**: Many surgical datasets (Cholec80, HeiChole, JIGSAWS, SAR-RARP50) require signing a data use agreement, Needs time 
2. **Frame extraction consistency**: Match extraction FPS/resolution pipeline to whatever is used for cataract-positive data to avoid the model learning dataset-artifact shortcuts instead of real surgical cues.
3. **Missing category alert**: There's a **scarcity of public datasets for other ophthalmic surgeries** (LASIK, vitrectomy, glaucoma surgery) — these are hardest negatives (Tier A from before) and likely need to be **manually scraped from YouTube/surgical education platforms** (e.g., Eyetube, Cybersight, ASCRS video archives) since no clean public dataset exists.

---
