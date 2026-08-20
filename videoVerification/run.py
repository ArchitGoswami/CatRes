import json
from videoVerification.videoValidator import (
    VideoValidationOrchestrator,
    VideoFileReader,
    FileValidator,
    FrameExtractor,
    TextDetector,
    DatasetSplitter
)


def runSingleVideoValidation():
    """Validate a single video file."""
    
    # Initialize orchestrator
    orchestrator = VideoValidationOrchestrator()
    
    # Path to your video file
    videoPath = "path/to/your/video.mp4"  # <-- CHANGE THIS
    
    # Run validation
    report = orchestrator.validateSingleVideo(
        videoPath,
        extractFrames=True,
        checkText=True,
        verbose=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(json.dumps(report.toDict(), indent=2))
    
    return report


def runFolderValidation():
    """Validate all videos in a folder."""
    
    orchestrator = VideoValidationOrchestrator()
    
    # Path to folder containing videos
    folderPath = "path/to/video/folder"  # <-- CHANGE THIS
    
    # Run validation
    results = orchestrator.validateFolder(
        folderPath,
        recursive=True,
        extractFrames=True,
        checkText=True,
        verbose=True
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total videos: {results['totalVideos']}")
    print(f"Passed: {results['passed']}")
    print(f"Warnings: {results['warnings']}")
    print(f"Failed: {results['failed']}")
    
    # Save detailed results to JSON
    with open("validationResults.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to: validationResults.json")
    
    return results


def runValidateAndSplit():
    """Validate videos and split into train/val/test sets."""
    
    orchestrator = VideoValidationOrchestrator()
    
    # Paths
    sourceFolderPath = "path/to/source/videos"  # <-- CHANGE THIS
    outputFolderPath = "path/to/output/dataset"  # <-- CHANGE THIS
    
    # Run validation and split
    result = orchestrator.validateAndSplit(
        folderPath=sourceFolderPath,
        outputDir=outputFolderPath,
        onlyPassed=True,
        includeWarnings=True,
        copyFiles=True,  # Set to True to actually copy files
        verbose=True
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total videos: {result['summary']['totalVideos']}")
    print(f"Passed validation: {result['summary']['passedValidation']}")
    print(f"Training set: {result['summary']['trainCount']}")
    print(f"Validation set: {result['summary']['validationCount']}")
    print(f"Test set: {result['summary']['testCount']}")
    
    # Save manifest
    with open("datasetManifest.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nManifest saved to: datasetManifest.json")
    
    return result


def runIndividualComponents():
    """Example of using individual components separately."""
    
    print("=" * 60)
    print("USING INDIVIDUAL COMPONENTS")
    print("=" * 60)
    
    # 1. Read video files from folder
    print("\n1. Reading video files...")
    reader = VideoFileReader()
    videoFiles = reader.readVideoFilesFromFolder(
        "path/to/videos",  # <-- CHANGE THIS
        recursive=True
    )
    print(f"   Found {len(videoFiles)} videos")
    
    if not videoFiles:
        print("   No videos found. Exiting.")
        return
    
    # 2. Validate file properties
    print("\n2. Validating file properties...")
    validator = FileValidator()
    
    for videoPath in videoFiles[:3]:  # Check first 3 videos
        print(f"\n   File: {videoPath}")
        
        extResult = validator.checkFileExtension(videoPath)
        print(f"   Extension: {extResult.status.value} - {extResult.message}")
        
        headerResult = validator.checkFileHeader(videoPath)
        print(f"   Header: {headerResult.status.value} - {headerResult.message}")
        
        sizeResult = validator.checkFileSizeBounds(videoPath)
        print(f"   Size: {sizeResult.status.value} - {sizeResult.message}")
    
    # 3. Extract frames
    print("\n3. Extracting frames...")
    extractor = FrameExtractor(numSegments=5, seed=42)
    
    frames, frameResult = extractor.extractSampleFrames(videoFiles[0])
    print(f"   {frameResult.message}")
    print(f"   Extracted {len([f for f in frames if f is not None])} frames")
    
    # 4. Analyze colors
    print("\n4. Analyzing colors...")
    colorResult = validator.analyzeColorValues(frames)
    print(f"   {colorResult.status.value} - {colorResult.message}")
    if colorResult.details.get("rgb"):
        rgb = colorResult.details["rgb"]
        print(f"   RGB means: R={rgb['redMean']}, G={rgb['greenMean']}, B={rgb['blueMean']}")
    
    # 5. Detect text
    print("\n5. Detecting text...")
    textDetector = TextDetector(useOcr=False)
    textResult = textDetector.checkFramesForText(frames)
    print(f"   {textResult.status.value} - {textResult.message}")
    
    # 6. Split dataset
    print("\n6. Splitting dataset...")
    splitter = DatasetSplitter(trainRatio=0.8, validationRatio=0.1, testRatio=0.1, seed=42)
    splits = splitter.splitVideoList(videoFiles)
    print(f"   Train: {splits['metadata']['trainCount']} videos")
    print(f"   Validation: {splits['metadata']['validationCount']} videos")
    print(f"   Test: {splits['metadata']['testCount']} videos")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VIDEO VALIDATION PIPELINE")
    print("=" * 60)
    print("\nSelect an option:")
    print("1. Validate a single video")
    print("2. Validate all videos in a folder")
    print("3. Validate and split into train/val/test")
    print("4. Run individual components demo")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        runSingleVideoValidation()
    elif choice == "2":
        runFolderValidation()
    elif choice == "3":
        runValidateAndSplit()
    elif choice == "4":
        runIndividualComponents()
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice. Please run again.")