# Video Validator Model

Model aims to validate to see if a video is a cataract surgery or not. This is to act as a precursor to prevent video analysis on irrelevant videos in Circlage.

## Summary of Functions

| Function | Class | Purpose |
|----------|-------|---------|
| **Orchestrator** | `VideoValidationOrchestrator` | Coordinates all validation steps |
| **Read Videos** | `VideoFileReader` | Reads video files from folders |
| **File Validation** | `FileValidator` | Extension, header, size, color checks |
| **Frame Extraction** | `FrameExtractor` | Gets 5 random frames from segments |
| **Text Detection** | `TextDetector` | Detects text presence in frames |
| **Dataset Split** | `DatasetSplitter` | 80/10/10 train/val/test split |

## Key Features

1. **CamelCase naming** throughout as requested
2. **Modular design** - each component can be used independently
3. **Detailed validation results** with status, message, and details
4. **Configurable thresholds** for all checks
5. **Multiple text detection methods** (edges, MSER, morphology, optional OCR)
6. **Statistical file size validation** using z-scores
7. **Reproducible splits** with seed parameter

## Installation Requirements

```bash
pip install opencv-python numpy
# Optional for OCR:
pip install pytesseract pillow
```

_____________________________________________________________________________________________________

Using JEPA to say what video is non-cataract surgery is out of distribution? May not need to train a new model at all.
Large models may already have enough non-relevant data to 

Reach out to Nisarg? Dr. Vedula will FIO, look out for email.

are there ANY new surgery datasets after 2024 (not surgery datasets)