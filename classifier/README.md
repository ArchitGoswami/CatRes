# Video Datasets for Hard Negative Mining

Here's a categorized list of **publicly available datasets** you can pull from, organized by which class they'd serve as negatives for.

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
