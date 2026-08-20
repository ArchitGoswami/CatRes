import os
import random
import shutil
import struct
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import cv2
import numpy as np

# Optional: For more advanced text detection
# import pytesseract
# from PIL import Image


class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationResult:
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class VideoValidationReport:
    filePath: str
    fileName: str
    fileExtensionCheck: ValidationResult
    fileHeaderCheck: ValidationResult
    fileSizeCheck: ValidationResult
    colorAnalysisCheck: ValidationResult
    frameExtractionCheck: ValidationResult
    textPresenceCheck: ValidationResult
    overallStatus: ValidationStatus
    
    def toDict(self) -> Dict:
        return {
            "filePath": self.filePath,
            "fileName": self.fileName,
            "fileExtensionCheck": {
                "status": self.fileExtensionCheck.status.value,
                "message": self.fileExtensionCheck.message,
                "details": self.fileExtensionCheck.details
            },
            "fileHeaderCheck": {
                "status": self.fileHeaderCheck.status.value,
                "message": self.fileHeaderCheck.message,
                "details": self.fileHeaderCheck.details
            },
            "fileSizeCheck": {
                "status": self.fileSizeCheck.status.value,
                "message": self.fileSizeCheck.message,
                "details": self.fileSizeCheck.details
            },
            "colorAnalysisCheck": {
                "status": self.colorAnalysisCheck.status.value,
                "message": self.colorAnalysisCheck.message,
                "details": self.colorAnalysisCheck.details
            },
            "frameExtractionCheck": {
                "status": self.frameExtractionCheck.status.value,
                "message": self.frameExtractionCheck.message,
                "details": self.frameExtractionCheck.details
            },
            "textPresenceCheck": {
                "status": self.textPresenceCheck.status.value,
                "message": self.textPresenceCheck.message,
                "details": self.textPresenceCheck.details
            },
            "overallStatus": self.overallStatus.value
        }


class VideoFileReader:
    """Function 2: Read video files from folders"""
    
    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    def __init__(self, supportedExtensions: Optional[set] = None):
        self.supportedExtensions = supportedExtensions or self.SUPPORTED_EXTENSIONS
    
    def readVideoFilesFromFolder(
        self, 
        folderPath: str, 
        recursive: bool = True
    ) -> List[str]:
        """
        Read all video files from a folder.
        
        Args:
            folderPath: Path to the folder containing videos
            recursive: Whether to search subdirectories
            
        Returns:
            List of absolute paths to video files
        """
        videoFiles = []
        folderPath = Path(folderPath)
        
        if not folderPath.exists():
            raise FileNotFoundError(f"Folder not found: {folderPath}")
        
        if not folderPath.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folderPath}")
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for filePath in folderPath.glob(pattern):
            if filePath.is_file() and filePath.suffix.lower() in self.supportedExtensions:
                videoFiles.append(str(filePath.absolute()))
        
        return sorted(videoFiles)
    
    def getVideoMetadata(self, videoPath: str) -> Dict[str, Any]:
        """Extract basic metadata from a video file."""
        cap = cv2.VideoCapture(videoPath)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {videoPath}")
        
        metadata = {
            "frameCount": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": None
        }
        
        if metadata["fps"] > 0:
            metadata["duration"] = metadata["frameCount"] / metadata["fps"]
        
        cap.release()
        return metadata


class FileValidator:
    """Function 3: File validation checks"""
    
    # Magic bytes for common video formats
    FILE_SIGNATURES = {
        '.mp4': [
            (b'\x00\x00\x00\x18ftypmp4', 0),   # MP4
            (b'\x00\x00\x00\x1cftypmp4', 0),   # MP4
            (b'\x00\x00\x00\x14ftypisom', 0),  # MP4 ISO
            (b'\x00\x00\x00\x18ftypisom', 0),  # MP4 ISO
            (b'\x00\x00\x00\x1cftypisom', 0),  # MP4 ISO
            (b'\x00\x00\x00 ftypisom', 0),     # MP4 ISO
            (b'ftyp', 4),                       # Generic MP4 (ftyp at offset 4)
        ],
        '.avi': [
            (b'RIFF', 0),  # AVI starts with RIFF
        ],
        '.mov': [
            (b'\x00\x00\x00\x14ftypqt', 0),    # QuickTime
            (b'moov', 4),                       # MOV alternate
            (b'ftyp', 4),                       # Generic QuickTime
        ],
        '.mkv': [
            (b'\x1a\x45\xdf\xa3', 0),          # Matroska/WebM
        ],
        '.webm': [
            (b'\x1a\x45\xdf\xa3', 0),          # WebM (same as Matroska)
        ],
        '.wmv': [
            (b'\x30\x26\xb2\x75\x8e\x66\xcf\x11', 0),  # ASF/WMV
        ],
        '.flv': [
            (b'FLV', 0),                       # Flash Video
        ],
    }
    
    # File size bounds based on surgical video characteristics
    # OphNet videos are typically 1-30 minutes, 720p-1080p
    DEFAULT_SIZE_BOUNDS = {
        "minSizeBytes": 100 * 1024,           # 100 KB minimum
        "maxSizeBytes": 10 * 1024 * 1024 * 1024,  # 10 GB maximum
        "expectedMeanBytes": 500 * 1024 * 1024,    # 500 MB expected mean
        "expectedStdBytes": 300 * 1024 * 1024,     # 300 MB standard deviation
        "zScoreThreshold": 3.0                     # Flag if beyond 3 std devs
    }
    
    # Expected color characteristics for surgical videos
    # Surgical videos typically have specific lighting and color profiles
    COLOR_EXPECTATIONS = {
        "minBrightness": 20,      # Not too dark
        "maxBrightness": 240,     # Not overexposed
        "expectedChannels": 3,    # RGB
        "minSaturation": 5,       # Some color present
        "maxRedDominance": 0.6,   # Red shouldn't dominate too much (blood consideration)
    }
    
    def __init__(
        self, 
        sizeBounds: Optional[Dict] = None,
        colorExpectations: Optional[Dict] = None
    ):
        self.sizeBounds = sizeBounds or self.DEFAULT_SIZE_BOUNDS
        self.colorExpectations = colorExpectations or self.COLOR_EXPECTATIONS
        self.observedFileSizes: List[int] = []
    
    def checkFileExtension(
        self, 
        filePath: str, 
        allowedExtensions: Optional[set] = None
    ) -> ValidationResult:
        """
        Check if file has a valid video extension.
        
        Args:
            filePath: Path to the file
            allowedExtensions: Set of allowed extensions (with dot)
            
        Returns:
            ValidationResult with status and details
        """
        if allowedExtensions is None:
            allowedExtensions = set(self.FILE_SIGNATURES.keys())
        
        extension = Path(filePath).suffix.lower()
        
        if extension in allowedExtensions:
            return ValidationResult(
                status=ValidationStatus.PASSED,
                message=f"Valid extension: {extension}",
                details={"extension": extension, "allowed": list(allowedExtensions)}
            )
        else:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Invalid extension: {extension}",
                details={"extension": extension, "allowed": list(allowedExtensions)}
            )
    
    def checkFileHeader(self, filePath: str) -> ValidationResult:
        """
        Verify file magic bytes match expected format.
        
        Args:
            filePath: Path to the file
            
        Returns:
            ValidationResult with status and details
        """
        extension = Path(filePath).suffix.lower()
        
        if extension not in self.FILE_SIGNATURES:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"No signature defined for extension: {extension}",
                details={"extension": extension}
            )
        
        try:
            with open(filePath, 'rb') as f:
                # Read first 32 bytes for signature checking
                headerBytes = f.read(32)
            
            signatures = self.FILE_SIGNATURES[extension]
            
            for signature, offset in signatures:
                if len(headerBytes) >= offset + len(signature):
                    if headerBytes[offset:offset + len(signature)].startswith(signature[:4]):
                        return ValidationResult(
                            status=ValidationStatus.PASSED,
                            message=f"File header matches {extension} format",
                            details={
                                "extension": extension,
                                "signatureMatched": signature[:8].hex(),
                                "headerBytes": headerBytes[:16].hex()
                            }
                        )
            
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"File header does not match {extension} format",
                details={
                    "extension": extension,
                    "headerBytes": headerBytes[:16].hex(),
                    "expectedSignatures": [sig[0][:8].hex() for sig in signatures]
                }
            )
            
        except IOError as e:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Cannot read file header: {str(e)}",
                details={"error": str(e)}
            )
    
    def checkFileSizeBounds(
        self, 
        filePath: str,
        updateObservations: bool = True
    ) -> ValidationResult:
        """
        Check if file size is within expected bounds.
        Uses running statistics if observations are available.
        
        Args:
            filePath: Path to the file
            updateObservations: Whether to add this file to observed sizes
            
        Returns:
            ValidationResult with status and details
        """
        try:
            fileSize = os.path.getsize(filePath)
        except OSError as e:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Cannot get file size: {str(e)}",
                details={"error": str(e)}
            )
        
        details = {
            "fileSizeBytes": fileSize,
            "fileSizeMB": round(fileSize / (1024 * 1024), 2)
        }
        
        # Check absolute bounds
        if fileSize < self.sizeBounds["minSizeBytes"]:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"File too small: {details['fileSizeMB']} MB",
                details={**details, "minSizeMB": self.sizeBounds["minSizeBytes"] / (1024 * 1024)}
            )
        
        if fileSize > self.sizeBounds["maxSizeBytes"]:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"File too large: {details['fileSizeMB']} MB",
                details={**details, "maxSizeMB": self.sizeBounds["maxSizeBytes"] / (1024 * 1024)}
            )
        
        # Calculate z-score if we have enough observations
        if len(self.observedFileSizes) >= 10:
            mean = statistics.mean(self.observedFileSizes)
            std = statistics.stdev(self.observedFileSizes)
            
            if std > 0:
                zScore = (fileSize - mean) / std
                details["zScore"] = round(zScore, 2)
                details["observedMeanMB"] = round(mean / (1024 * 1024), 2)
                details["observedStdMB"] = round(std / (1024 * 1024), 2)
                
                if abs(zScore) > self.sizeBounds["zScoreThreshold"]:
                    return ValidationResult(
                        status=ValidationStatus.WARNING,
                        message=f"File size is unusual (z-score: {zScore:.2f})",
                        details=details
                    )
        else:
            # Use default expected values
            zScore = (fileSize - self.sizeBounds["expectedMeanBytes"]) / self.sizeBounds["expectedStdBytes"]
            details["zScore"] = round(zScore, 2)
            
            if abs(zScore) > self.sizeBounds["zScoreThreshold"]:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"File size deviates from expected (z-score: {zScore:.2f})",
                    details=details
                )
        
        # Update observations
        if updateObservations:
            self.observedFileSizes.append(fileSize)
        
        return ValidationResult(
            status=ValidationStatus.PASSED,
            message=f"File size within bounds: {details['fileSizeMB']} MB",
            details=details
        )
    
    def analyzeColorValues(
        self, 
        frames: List[np.ndarray]
    ) -> ValidationResult:
        """
        Analyze color characteristics of video frames.
        
        Args:
            frames: List of frames (numpy arrays) to analyze
            
        Returns:
            ValidationResult with color analysis details
        """
        if not frames:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message="No frames provided for color analysis",
                details={}
            )
        
        allRedMeans = []
        allGreenMeans = []
        allBlueMeans = []
        allBrightness = []
        allSaturation = []
        channelCounts = []
        
        for frame in frames:
            if frame is None:
                continue
            
            # Check number of channels
            if len(frame.shape) == 2:
                channelCounts.append(1)  # Grayscale
                allBrightness.append(np.mean(frame))
            elif len(frame.shape) == 3:
                channelCounts.append(frame.shape[2])
                
                if frame.shape[2] == 3:
                    # BGR format from OpenCV
                    blue, green, red = cv2.split(frame)
                    
                    allRedMeans.append(np.mean(red))
                    allGreenMeans.append(np.mean(green))
                    allBlueMeans.append(np.mean(blue))
                    
                    # Calculate brightness (simple average)
                    brightness = np.mean(frame)
                    allBrightness.append(brightness)
                    
                    # Calculate saturation from HSV
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    saturation = np.mean(hsv[:, :, 1])
                    allSaturation.append(saturation)
        
        if not allBrightness:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message="Could not analyze any frames",
                details={}
            )
        
        # Compile statistics
        details = {
            "frameCount": len(frames),
            "analyzedFrames": len(allBrightness),
            "channels": {
                "mode": max(set(channelCounts), key=channelCounts.count) if channelCounts else None,
                "counts": channelCounts
            },
            "brightness": {
                "mean": round(np.mean(allBrightness), 2),
                "std": round(np.std(allBrightness), 2),
                "min": round(np.min(allBrightness), 2),
                "max": round(np.max(allBrightness), 2)
            }
        }
        
        if allRedMeans:
            totalColorMean = np.mean(allRedMeans) + np.mean(allGreenMeans) + np.mean(allBlueMeans)
            redDominance = np.mean(allRedMeans) / totalColorMean if totalColorMean > 0 else 0
            
            details["rgb"] = {
                "redMean": round(np.mean(allRedMeans), 2),
                "greenMean": round(np.mean(allGreenMeans), 2),
                "blueMean": round(np.mean(allBlueMeans), 2),
                "redDominance": round(redDominance, 3)
            }
            
            details["saturation"] = {
                "mean": round(np.mean(allSaturation), 2),
                "std": round(np.std(allSaturation), 2)
            }
        
        # Validation checks
        warnings = []
        failures = []
        
        # Check brightness
        avgBrightness = details["brightness"]["mean"]
        if avgBrightness < self.colorExpectations["minBrightness"]:
            failures.append(f"Image too dark (brightness: {avgBrightness})")
        elif avgBrightness > self.colorExpectations["maxBrightness"]:
            failures.append(f"Image overexposed (brightness: {avgBrightness})")
        
        # Check channels
        if details["channels"]["mode"] != self.colorExpectations["expectedChannels"]:
            warnings.append(f"Unexpected channel count: {details['channels']['mode']}")
        
        # Check saturation (if available)
        if "saturation" in details:
            if details["saturation"]["mean"] < self.colorExpectations["minSaturation"]:
                warnings.append(f"Very low saturation: {details['saturation']['mean']}")
        
        # Check red dominance (surgical videos may have blood)
        if "rgb" in details:
            if details["rgb"]["redDominance"] > self.colorExpectations["maxRedDominance"]:
                warnings.append(f"High red dominance: {details['rgb']['redDominance']}")
        
        if failures:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message="; ".join(failures),
                details=details
            )
        elif warnings:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message="; ".join(warnings),
                details=details
            )
        else:
            return ValidationResult(
                status=ValidationStatus.PASSED,
                message="Color analysis passed",
                details=details
            )


class FrameExtractor:
    """Function 4: Extract sample frames from video segments"""
    
    def __init__(self, numSegments: int = 5, seed: Optional[int] = None):
        self.numSegments = numSegments
        if seed is not None:
            random.seed(seed)
    
    def extractSampleFrames(
        self, 
        videoPath: str,
        numSegments: Optional[int] = None
    ) -> Tuple[List[np.ndarray], ValidationResult]:
        """
        Extract random frames from each segment of a video.
        
        Divides video into N segments and extracts one random frame
        from each segment.
        
        Args:
            videoPath: Path to the video file
            numSegments: Number of segments (default: 5)
            
        Returns:
            Tuple of (list of frames, ValidationResult)
        """
        if numSegments is None:
            numSegments = self.numSegments
        
        frames = []
        frameIndices = []
        
        cap = cv2.VideoCapture(videoPath)
        
        if not cap.isOpened():
            return [], ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Cannot open video: {videoPath}",
                details={"error": "Failed to open video capture"}
            )
        
        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if totalFrames < numSegments:
            cap.release()
            return [], ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Video too short: {totalFrames} frames for {numSegments} segments",
                details={"totalFrames": totalFrames, "requestedSegments": numSegments}
            )
        
        # Calculate segment boundaries
        segmentSize = totalFrames // numSegments
        
        for i in range(numSegments):
            segmentStart = i * segmentSize
            segmentEnd = (i + 1) * segmentSize if i < numSegments - 1 else totalFrames
            
            # Random position within segment
            frameIndex = random.randint(segmentStart, segmentEnd - 1)
            frameIndices.append(frameIndex)
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
            else:
                frames.append(None)
        
        cap.release()
        
        successCount = sum(1 for f in frames if f is not None)
        
        if successCount == 0:
            return frames, ValidationResult(
                status=ValidationStatus.FAILED,
                message="Failed to extract any frames",
                details={
                    "totalFrames": totalFrames,
                    "attemptedIndices": frameIndices,
                    "successCount": 0
                }
            )
        elif successCount < numSegments:
            return frames, ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Extracted {successCount}/{numSegments} frames",
                details={
                    "totalFrames": totalFrames,
                    "frameIndices": frameIndices,
                    "successCount": successCount,
                    "fps": fps
                }
            )
        else:
            return frames, ValidationResult(
                status=ValidationStatus.PASSED,
                message=f"Successfully extracted {numSegments} frames",
                details={
                    "totalFrames": totalFrames,
                    "frameIndices": frameIndices,
                    "successCount": successCount,
                    "fps": fps,
                    "segmentSize": segmentSize
                }
            )


class TextDetector:
    """Function 5: Detect text presence in frames"""
    
    def __init__(
        self, 
        useOcr: bool = False,
        edgeThreshold: float = 0.1,
        textRegionMinArea: int = 100
    ):
        """
        Initialize text detector.
        
        Args:
            useOcr: Whether to use OCR (requires pytesseract)
            edgeThreshold: Threshold for edge-based text detection
            textRegionMinArea: Minimum area for potential text regions
        """
        self.useOcr = useOcr
        self.edgeThreshold = edgeThreshold
        self.textRegionMinArea = textRegionMinArea
    
    def detectTextInFrame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect potential text presence in a single frame.
        
        Uses multiple heuristics:
        1. Edge density analysis (text has high edge density)
        2. Connected component analysis
        3. Optional OCR for confirmation
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with detection results
        """
        if frame is None:
            return {
                "hasText": False,
                "confidence": 0.0,
                "method": "none",
                "error": "Frame is None"
            }
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        result = {
            "hasText": False,
            "confidence": 0.0,
            "regions": [],
            "methods": {}
        }
        
        # Method 1: Edge-based detection
        edgeResult = self.detectTextByEdges(gray)
        result["methods"]["edges"] = edgeResult
        
        # Method 2: MSER (Maximally Stable Extremal Regions)
        mserResult = self.detectTextByMser(gray)
        result["methods"]["mser"] = mserResult
        
        # Method 3: Morphological text detection
        morphResult = self.detectTextByMorphology(gray)
        result["methods"]["morphology"] = morphResult
        
        # Combine results
        confidences = [
            edgeResult.get("confidence", 0),
            mserResult.get("confidence", 0),
            morphResult.get("confidence", 0)
        ]
        
        avgConfidence = np.mean(confidences)
        maxConfidence = np.max(confidences)
        
        # Text is likely present if multiple methods agree
        methodsDetected = sum(1 for c in confidences if c > 0.3)
        
        result["confidence"] = round(float(avgConfidence), 3)
        result["maxConfidence"] = round(float(maxConfidence), 3)
        result["methodsAgreeing"] = methodsDetected
        result["hasText"] = methodsDetected >= 2 or maxConfidence > 0.6
        
        # Optional: OCR confirmation
        if self.useOcr and result["hasText"]:
            ocrResult = self.detectTextByOcr(frame)
            result["methods"]["ocr"] = ocrResult
            if ocrResult.get("textFound"):
                result["ocrText"] = ocrResult.get("text", "")
        
        return result
    
    def detectTextByEdges(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect text using edge density analysis."""
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        totalPixels = gray.shape[0] * gray.shape[1]
        edgePixels = np.sum(edges > 0)
        edgeDensity = edgePixels / totalPixels
        
        # Text typically has moderate edge density in localized regions
        # Analyze edge density in grid cells
        gridSize = 8
        cellHeight = gray.shape[0] // gridSize
        cellWidth = gray.shape[1] // gridSize
        
        highDensityCells = 0
        textLikeCells = 0
        
        for i in range(gridSize):
            for j in range(gridSize):
                cellEdges = edges[
                    i * cellHeight:(i + 1) * cellHeight,
                    j * cellWidth:(j + 1) * cellWidth
                ]
                cellDensity = np.sum(cellEdges > 0) / (cellHeight * cellWidth)
                
                if 0.05 < cellDensity < 0.3:  # Text-like edge density
                    textLikeCells += 1
                if cellDensity > 0.1:
                    highDensityCells += 1
        
        # Confidence based on text-like cells
        confidence = min(textLikeCells / (gridSize * gridSize * 0.3), 1.0)
        
        return {
            "edgeDensity": round(float(edgeDensity), 4),
            "textLikeCells": textLikeCells,
            "highDensityCells": highDensityCells,
            "confidence": round(float(confidence), 3)
        }
    
    def detectTextByMser(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect text using MSER (Maximally Stable Extremal Regions)."""
        mser = cv2.MSER_create()
        mser.setMinArea(60)
        mser.setMaxArea(14400)
        
        try:
            regions, _ = mser.detectRegions(gray)
        except cv2.error:
            return {"confidence": 0, "regionCount": 0, "error": "MSER detection failed"}
        
        # Filter regions by aspect ratio (text characters are usually tall/narrow or square)
        textLikeRegions = 0
        
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            aspectRatio = w / h if h > 0 else 0
            area = w * h
            
            # Text-like characteristics
            if 0.1 < aspectRatio < 10 and area > self.textRegionMinArea:
                textLikeRegions += 1
        
        # Confidence based on number of text-like regions
        # Surgical videos typically have timestamps/overlays with 5-50 characters
        confidence = min(textLikeRegions / 50, 1.0) if textLikeRegions > 3 else 0
        
        return {
            "totalRegions": len(regions),
            "textLikeRegions": textLikeRegions,
            "confidence": round(float(confidence), 3)
        }
    
    def detectTextByMorphology(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect text using morphological operations."""
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text characters
        kernelHorizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernelVertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        
        # Dilate to connect characters
        dilated = cv2.dilate(binary, kernelHorizontal, iterations=1)
        dilated = cv2.dilate(dilated, kernelVertical, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        textLikeContours = 0
        textRegions = []
        
        imageArea = gray.shape[0] * gray.shape[1]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspectRatio = w / h if h > 0 else 0
            
            # Text regions are typically wide and short (for horizontal text)
            # or narrow and tall (for vertical text)
            # Also, they shouldn't be too large (not the whole image)
            relativeArea = area / imageArea
            
            if (aspectRatio > 2 and 0.001 < relativeArea < 0.1) or \
               (0.5 < aspectRatio < 3 and 0.0005 < relativeArea < 0.05):
                textLikeContours += 1
                textRegions.append({
                    "x": x, "y": y, "width": w, "height": h,
                    "aspectRatio": round(aspectRatio, 2)
                })
        
        confidence = min(textLikeContours / 10, 1.0) if textLikeContours > 0 else 0
        
        return {
            "totalContours": len(contours),
            "textLikeContours": textLikeContours,
            "textRegions": textRegions[:5],  # Return top 5 regions
            "confidence": round(float(confidence), 3)
        }
    
    def detectTextByOcr(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect text using OCR (requires pytesseract)."""
        try:
            import pytesseract
            from PIL import Image
            
            # Convert to PIL Image
            if len(frame.shape) == 3:
                frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frameRgb = frame
            
            pilImage = Image.fromarray(frameRgb)
            
            # Run OCR
            text = pytesseract.image_to_string(pilImage, timeout=5)
            text = text.strip()
            
            return {
                "textFound": len(text) > 0,
                "text": text[:200] if text else "",  # Limit text length
                "charCount": len(text),
                "confidence": min(len(text) / 20, 1.0) if text else 0
            }
        except ImportError:
            return {
                "textFound": False,
                "error": "pytesseract not installed",
                "confidence": 0
            }
        except Exception as e:
            return {
                "textFound": False,
                "error": str(e),
                "confidence": 0
            }
    
    def checkFramesForText(
        self, 
        frames: List[np.ndarray]
    ) -> ValidationResult:
        """
        Check multiple frames for text presence.
        
        Args:
            frames: List of frames to analyze
            
        Returns:
            ValidationResult with text detection summary
        """
        if not frames:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                message="No frames to analyze",
                details={}
            )
        
        frameResults = []
        framesWithText = 0
        
        for i, frame in enumerate(frames):
            if frame is None:
                frameResults.append({"frameIndex": i, "error": "Frame is None"})
                continue
            
            result = self.detectTextInFrame(frame)
            result["frameIndex"] = i
            frameResults.append(result)
            
            if result.get("hasText", False):
                framesWithText += 1
        
        validFrames = sum(1 for f in frames if f is not None)
        textRatio = framesWithText / validFrames if validFrames > 0 else 0
        
        details = {
            "totalFrames": len(frames),
            "validFrames": validFrames,
            "framesWithText": framesWithText,
            "textRatio": round(textRatio, 3),
            "frameResults": frameResults
        }
        
        # Determine status based on text presence
        # Some text is expected (timestamps, etc.), but too much might indicate wrong content
        if textRatio > 0.8:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"High text presence: {framesWithText}/{validFrames} frames",
                details=details
            )
        elif textRatio > 0:
            return ValidationResult(
                status=ValidationStatus.PASSED,
                message=f"Text detected in {framesWithText}/{validFrames} frames (normal for surgical videos)",
                details=details
            )
        else:
            return ValidationResult(
                status=ValidationStatus.PASSED,
                message="No significant text detected",
                details=details
            )

# TODO: NOT IMP RN
class DatasetSplitter:
    """Function 7: Split videos into train/validation/test sets"""
    
    def __init__(
        self,
        trainRatio: float = 0.8,
        validationRatio: float = 0.1,
        testRatio: float = 0.1,
        seed: Optional[int] = 42
    ):
        """
        Initialize dataset splitter.
        
        Args:
            trainRatio: Proportion for training set
            validationRatio: Proportion for validation set
            testRatio: Proportion for test set
            seed: Random seed for reproducibility
        """
        if abs(trainRatio + validationRatio + testRatio - 1.0) > 0.001:
            raise ValueError("Ratios must sum to 1.0")
        
        self.trainRatio = trainRatio
        self.validationRatio = validationRatio
        self.testRatio = testRatio
        self.seed = seed
    
    def splitVideoList(
        self,
        videoFiles: List[str],
        shuffle: bool = True
    ) -> Dict[str, List[str]]:
        """
        Split a list of video files into train/validation/test sets.
        
        Args:
            videoFiles: List of video file paths
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Dictionary with 'train', 'validation', 'test' keys
        """
        if self.seed is not None:
            random.seed(self.seed)
        
        files = videoFiles.copy()
        
        if shuffle:
            random.shuffle(files)
        
        totalCount = len(files)
        trainCount = int(totalCount * self.trainRatio)
        validationCount = int(totalCount * self.validationRatio)
        
        trainFiles = files[:trainCount]
        validationFiles = files[trainCount:trainCount + validationCount]
        testFiles = files[trainCount + validationCount:]
        
        return {
            "train": trainFiles,
            "validation": validationFiles,
            "test": testFiles,
            "metadata": {
                "totalCount": totalCount,
                "trainCount": len(trainFiles),
                "validationCount": len(validationFiles),
                "testCount": len(testFiles),
                "trainRatio": round(len(trainFiles) / totalCount, 3) if totalCount > 0 else 0,
                "validationRatio": round(len(validationFiles) / totalCount, 3) if totalCount > 0 else 0,
                "testRatio": round(len(testFiles) / totalCount, 3) if totalCount > 0 else 0,
                "seed": self.seed
            }
        }
    
    def splitAndCopyFiles(
        self,
        videoFiles: List[str],
        outputDir: str,
        shuffle: bool = True,
        copyFiles: bool = True,
        createSymlinks: bool = False
    ) -> Dict[str, Any]:
        """
        Split videos and organize into directory structure.
        
        Creates:
            outputDir/
                train/
                validation/
                test/
        
        Args:
            videoFiles: List of video file paths
            outputDir: Base output directory
            shuffle: Whether to shuffle before splitting
            copyFiles: Whether to copy files (False = just return split)
            createSymlinks: Create symlinks instead of copying (Linux/Mac)
            
        Returns:
            Dictionary with split results and file locations
        """
        splits = self.splitVideoList(videoFiles, shuffle)
        
        outputPath = Path(outputDir)
        
        result = {
            "outputDir": str(outputPath),
            "splits": {},
            "metadata": splits["metadata"]
        }
        
        for splitName in ["train", "validation", "test"]:
            splitDir = outputPath / splitName
            splitDir.mkdir(parents=True, exist_ok=True)
            
            splitFiles = []
            
            for sourcePath in splits[splitName]:
                fileName = Path(sourcePath).name
                destPath = splitDir / fileName
                
                if copyFiles:
                    if createSymlinks:
                        if not destPath.exists():
                            destPath.symlink_to(Path(sourcePath).absolute())
                    else:
                        if not destPath.exists():
                            shutil.copy2(sourcePath, destPath)
                
                splitFiles.append({
                    "source": sourcePath,
                    "destination": str(destPath)
                })
            
            result["splits"][splitName] = {
                "directory": str(splitDir),
                "count": len(splitFiles),
                "files": splitFiles
            }
        
        return result
    
    def createSplitManifest(
        self,
        videoFiles: List[str],
        outputPath: str,
        shuffle: bool = True
    ) -> str:
        """
        Create a manifest file documenting the split.
        
        Args:
            videoFiles: List of video file paths
            outputPath: Path for the manifest file
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Path to created manifest file
        """
        import json
        from datetime import datetime
        
        splits = self.splitVideoList(videoFiles, shuffle)
        
        manifest = {
            "createdAt": datetime.now().isoformat(),
            "config": {
                "trainRatio": self.trainRatio,
                "validationRatio": self.validationRatio,
                "testRatio": self.testRatio,
                "seed": self.seed,
                "shuffled": shuffle
            },
            "metadata": splits["metadata"],
            "splits": {
                "train": splits["train"],
                "validation": splits["validation"],
                "test": splits["test"]
            }
        }
        
        with open(outputPath, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return outputPath


class VideoValidationOrchestrator:
    """Function 1: Orchestrator that runs all validation functions"""
    
    def __init__(
        self,
        fileReader: Optional[VideoFileReader] = None,
        fileValidator: Optional[FileValidator] = None,
        frameExtractor: Optional[FrameExtractor] = None,
        textDetector: Optional[TextDetector] = None,
        datasetSplitter: Optional[DatasetSplitter] = None
    ):
        """
        Initialize orchestrator with component instances.
        
        Args:
            fileReader: VideoFileReader instance
            fileValidator: FileValidator instance
            frameExtractor: FrameExtractor instance
            textDetector: TextDetector instance
            datasetSplitter: DatasetSplitter instance
        """
        self.fileReader = fileReader or VideoFileReader()
        self.fileValidator = fileValidator or FileValidator()
        self.frameExtractor = frameExtractor or FrameExtractor()
        self.textDetector = textDetector or TextDetector()
        self.datasetSplitter = datasetSplitter or DatasetSplitter()
    
    def validateSingleVideo(
        self,
        videoPath: str,
        extractFrames: bool = True,
        checkText: bool = True,
        verbose: bool = False
    ) -> VideoValidationReport:
        """
        Run full validation pipeline on a single video.
        
        Args:
            videoPath: Path to video file
            extractFrames: Whether to extract and analyze frames
            checkText: Whether to check for text in frames
            verbose: Print progress messages
            
        Returns:
            VideoValidationReport with all check results
        """
        if verbose:
            print(f"Validating: {videoPath}")
        
        fileName = Path(videoPath).name
        
        # Step 1: File extension check
        if verbose:
            print("  - Checking file extension...")
        extensionResult = self.fileValidator.checkFileExtension(videoPath)
        
        # Step 2: File header check
        if verbose:
            print("  - Checking file header...")
        headerResult = self.fileValidator.checkFileHeader(videoPath)
        
        # Step 3: File size check
        if verbose:
            print("  - Checking file size...")
        sizeResult = self.fileValidator.checkFileSizeBounds(videoPath)
        
        # Step 4: Extract frames
        frames = []
        frameResult = ValidationResult(
            status=ValidationStatus.PASSED,
            message="Frame extraction skipped",
            details={}
        )
        
        if extractFrames:
            if verbose:
                print("  - Extracting sample frames...")
            frames, frameResult = self.frameExtractor.extractSampleFrames(videoPath)
        
        # Step 5: Color analysis
        colorResult = ValidationResult(
            status=ValidationStatus.PASSED,
            message="Color analysis skipped",
            details={}
        )
        
        if frames:
            if verbose:
                print("  - Analyzing color values...")
            colorResult = self.fileValidator.analyzeColorValues(frames)
        
        # Step 6: Text detection
        textResult = ValidationResult(
            status=ValidationStatus.PASSED,
            message="Text detection skipped",
            details={}
        )
        
        if checkText and frames:
            if verbose:
                print("  - Detecting text in frames...")
            textResult = self.textDetector.checkFramesForText(frames)
        
        # Determine overall status
        allResults = [
            extensionResult,
            headerResult,
            sizeResult,
            frameResult,
            colorResult,
            textResult
        ]
        
        if any(r.status == ValidationStatus.FAILED for r in allResults):
            overallStatus = ValidationStatus.FAILED
        elif any(r.status == ValidationStatus.WARNING for r in allResults):
            overallStatus = ValidationStatus.WARNING
        else:
            overallStatus = ValidationStatus.PASSED
        
        if verbose:
            print(f"  - Overall status: {overallStatus.value}")
        
        return VideoValidationReport(
            filePath=videoPath,
            fileName=fileName,
            fileExtensionCheck=extensionResult,
            fileHeaderCheck=headerResult,
            fileSizeCheck=sizeResult,
            colorAnalysisCheck=colorResult,
            frameExtractionCheck=frameResult,
            textPresenceCheck=textResult,
            overallStatus=overallStatus
        )
    
    def validateFolder(
        self,
        folderPath: str,
        recursive: bool = True,
        extractFrames: bool = True,
        checkText: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Validate all videos in a folder.
        
        Args:
            folderPath: Path to folder containing videos
            recursive: Search subdirectories
            extractFrames: Extract and analyze frames
            checkText: Check for text in frames
            verbose: Print progress
            
        Returns:
            Dictionary with validation results for all videos
        """
        if verbose:
            print(f"Scanning folder: {folderPath}")
        
        videoFiles = self.fileReader.readVideoFilesFromFolder(folderPath, recursive)
        
        if verbose:
            print(f"Found {len(videoFiles)} video files")
        
        results = {
            "folderPath": folderPath,
            "totalVideos": len(videoFiles),
            "passed": 0,
            "warnings": 0,
            "failed": 0,
            "reports": []
        }
        
        for i, videoPath in enumerate(videoFiles):
            if verbose:
                print(f"\n[{i + 1}/{len(videoFiles)}]")
            
            report = self.validateSingleVideo(
                videoPath,
                extractFrames=extractFrames,
                checkText=checkText,
                verbose=verbose
            )
            
            results["reports"].append(report.toDict())
            
            if report.overallStatus == ValidationStatus.PASSED:
                results["passed"] += 1
            elif report.overallStatus == ValidationStatus.WARNING:
                results["warnings"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    def validateAndSplit(
        self,
        folderPath: str,
        outputDir: str,
        onlyPassed: bool = True,
        includeWarnings: bool = True,
        copyFiles: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Validate videos and split passing ones into train/val/test.
        
        Args:
            folderPath: Source folder with videos
            outputDir: Output directory for splits
            onlyPassed: Only include passed videos
            includeWarnings: Include videos with warnings
            copyFiles: Copy files to output directories
            verbose: Print progress
            
        Returns:
            Dictionary with validation and split results
        """
        # Validate all videos
        validationResults = self.validateFolder(
            folderPath,
            extractFrames=True,
            checkText=True,
            verbose=verbose
        )
        
        # Filter videos based on status
        passingVideos = []
        
        for report in validationResults["reports"]:
            status = report["overallStatus"]
            
            if status == "passed":
                passingVideos.append(report["filePath"])
            elif status == "warning" and includeWarnings:
                passingVideos.append(report["filePath"])
        
        if verbose:
            print(f"\nVideos passing validation: {len(passingVideos)}")
        
        # Split videos
        splitResult = self.datasetSplitter.splitAndCopyFiles(
            passingVideos,
            outputDir,
            copyFiles=copyFiles
        )
        
        return {
            "validation": validationResults,
            "split": splitResult,
            "summary": {
                "totalVideos": validationResults["totalVideos"],
                "passedValidation": len(passingVideos),
                "trainCount": splitResult["metadata"]["trainCount"],
                "validationCount": splitResult["metadata"]["validationCount"],
                "testCount": splitResult["metadata"]["testCount"]
            }
        }


# =============================================================================
# Example Usage
# =============================================================================

def main():
    """Example usage of the video validation pipeline."""
    
    # Initialize orchestrator with default settings
    orchestrator = VideoValidationOrchestrator()
    
    # Example 1: Validate a single video
    print("=" * 60)
    print("Example 1: Single Video Validation")
    print("=" * 60)
    
    # Uncomment to test with a real video:
    # report = orchestrator.validateSingleVideo(
    #     "path/to/your/video.mp4",
    #     verbose=True
    # )
    # print(json.dumps(report.toDict(), indent=2))
    
    # Example 2: Validate all videos in a folder
    print("\n" + "=" * 60)
    print("Example 2: Folder Validation")
    print("=" * 60)
    
    # Uncomment to test with a real folder:
    # results = orchestrator.validateFolder(
    #     "path/to/video/folder",
    #     recursive=True,
    #     verbose=True
    # )
    # print(f"Passed: {results['passed']}")
    # print(f"Warnings: {results['warnings']}")
    # print(f"Failed: {results['failed']}")
    
    # Example 3: Validate and split into train/val/test
    print("\n" + "=" * 60)
    print("Example 3: Validate and Split Dataset")
    print("=" * 60)
    
    # Uncomment to test with real data:
    # result = orchestrator.validateAndSplit(
    #     folderPath="path/to/source/videos",
    #     outputDir="path/to/output/dataset",
    #     onlyPassed=True,
    #     includeWarnings=True,
    #     copyFiles=True,
    #     verbose=True
    # )
    # print(f"Summary: {result['summary']}")
    
    # Example 4: Custom configuration
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)
    
    customOrchestrator = VideoValidationOrchestrator(
        fileValidator=FileValidator(
            sizeBounds={
                "minSizeBytes": 1 * 1024 * 1024,      # 1 MB minimum
                "maxSizeBytes": 5 * 1024 * 1024 * 1024,  # 5 GB maximum
                "expectedMeanBytes": 200 * 1024 * 1024,
                "expectedStdBytes": 100 * 1024 * 1024,
                "zScoreThreshold": 2.5
            }
        ),
        frameExtractor=FrameExtractor(numSegments=10, seed=42),
        textDetector=TextDetector(useOcr=False),
        datasetSplitter=DatasetSplitter(
            trainRatio=0.7,
            validationRatio=0.15,
            testRatio=0.15,
            seed=123
        )
    )
    
    print("Custom orchestrator created with:")
    print("  - File size bounds: 1MB - 5GB")
    print("  - Frame extraction: 10 segments")
    print("  - Dataset split: 70/15/15")
    
    print("\n" + "=" * 60)
    print("Pipeline ready for use!")
    print("=" * 60)


if __name__ == "__main__":
    main()