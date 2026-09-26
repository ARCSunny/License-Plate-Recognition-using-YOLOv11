# License Plate Detection and OCR

https://github.com/user-attachments/assets/7eae1b02-54ac-4515-8e2b-a0b127de005b

A computer vision project for **license plate detection, tracking, and text recognition** using **YOLO11**, **ByteTrack**, **EasyOCR**, and **OpenCV**.

The project is implemented as a Google Colab/Jupyter notebook and is designed to process vehicle videos, detect license plates, track them across frames, and extract the detected plate text.

## Features

- License plate object detection with **YOLO11**
- Training on a license-plate dataset downloaded from **Roboflow**
- GPU/CPU availability checks and memory monitoring
- Model evaluation using:
  - mAP@50
  - mAP@50-95
- License plate text recognition using **EasyOCR**
- Multi-object tracking using **ByteTrack**
- Frame-by-frame OCR with configurable frequency
- Automatic aggregation of OCR results for each tracked plate
- Annotated MP4 video output
- Export of the trained YOLO weights

## Project Structure

```
.
├── LicensePlateDetection.ipynb   # Main training, evaluation, OCR and tracking notebook
├── best.pt                       # Trained YOLO11 model weights
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO11 / Ultralytics | License plate detection |
| Roboflow | Dataset download and management |
| EasyOCR | License plate text recognition |
| OpenCV | Image/video processing |
| ByteTrack | Object tracking |
| PyTorch | Deep learning framework |
| NumPy | Numerical operations |
| PyYAML | Dataset configuration |
| Pillow | Image processing |

## Dataset

The notebook downloads the license-plate dataset from Roboflow using the Roboflow Python SDK.

The dataset configuration is adjusted to use the following YOLO directory structure:

```
lp_dataset_raw/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

You will need a **Roboflow API key** to download the dataset.

The notebook prompts for the key securely:

```
from getpass import getpass

api_key = getpass("API Key: ")
```

Do **not** commit your Roboflow API key or other credentials to GitHub.

## Model Training

The project uses the YOLO11 small model:

```
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
```

Training is configured with:

```
results = model.train(
    data=str(yaml_path),
    time=2.0,
    imgsz=640,
    batch=16,
    patience=15,
    device=0,
    amp=True,
    cache=False,
    project="/content/runs",
    name="license_plate",
    exist_ok=True,
    workers=2,
    plots=True,
)
```

### Main Training Parameters

- **Image size:** 640 × 640
- **Batch size:** 16
- **Training time:** up to 2 hours
- **Early stopping patience:** 15
- **Device:** GPU (`device=0`)
- **Automatic mixed precision:** enabled
- **Model:** YOLO11s

Adjust these parameters according to your available GPU memory and dataset size.

## Model Evaluation

After training, the best checkpoint is loaded and evaluated:

```
best = "/content/runs/license_plate/weights/best.pt"

model = YOLO(best)
metrics = model.val(data=str(yaml_path), device=0)

print(f"mAP@50: {metrics.box.map50:.3f}")
print(f"mAP@50-95: {metrics.box.map:.3f}")
```

The training results are also saved as:

```
/content/runs/license_plate/results.png
```

## OCR

After license plates are detected, the detected plate regions are passed to EasyOCR.

```
reader = easyocr.Reader(
    ["en"],
    gpu=torch.cuda.is_available()
)
```

The OCR pipeline:

1. Crops the detected license plate.
2. Converts the crop to grayscale.
3. Upscales the image.
4. Runs EasyOCR.
5. Selects the highest-confidence OCR result.
6. Removes non-alphanumeric characters.
7. Converts the result to uppercase.

Example:

```
Original OCR result: "DHA-1234"
Processed result:    "DHA1234"
```

## Video Tracking

The trained detector is used with **ByteTrack** to maintain object identities across video frames:

```
for r in model.track(
    source=video_path,
    stream=True,
    conf=0.35,
    iou=0.5,
    imgsz=640,
    persist=True,
    tracker="bytetrack.yaml",
    verbose=False
):
    ...
```

OCR is not performed on every frame. By default, it runs every third frame:

```
OCR_EVERY_N_FRAMES = 3
```

The OCR results for each tracking ID are stored and the most frequently recognized plate text is displayed for that tracked object.

## Output

The notebook generates an annotated video containing:

- Bounding boxes around detected license plates
- Detection confidence
- Recognized license plate text
- Tracking IDs internally used to maintain plate identity

The final re-encoded video is:

```
/content/output/result.mp4
```

The trained model is saved as:

```
/content/runs/license_plate/weights/best.pt
```
### Image Size

```
imgsz=640
```

Larger images can potentially improve detection of small plates but require more GPU memory.

### OCR Frequency

```
OCR_EVERY_N_FRAMES = 3
```

A lower value performs OCR more frequently, while a higher value reduces OCR computation.

## Performance Considerations

For better performance:

- Use a CUDA-enabled GPU when available.
- Reduce `batch` if GPU memory is insufficient.
- Reduce `imgsz` for faster inference.
- Increase `OCR_EVERY_N_FRAMES` to reduce OCR overhead.
- Avoid caching the entire dataset when system memory is limited.
- Keep video resolution reasonable for real-time or near-real-time processing.

## Limitations

- OCR accuracy depends heavily on plate image quality, lighting, viewing angle, motion blur, and character visibility.
- The current OCR configuration uses English characters only.
- The project is primarily demonstrated through a notebook rather than a standalone application.
- Google Colab-specific upload/download functionality is used in the video-processing workflow.
- Detection and OCR performance depends on the dataset used for training.
- The included `best.pt` weights should be evaluated on your target environment and data before production deployment.

## Acknowledgements

This project uses:

- **Ultralytics YOLO** for object detection
- **Roboflow** for dataset management
- **EasyOCR** for optical character recognition
- **OpenCV** for computer vision and video processing
- **ByteTrack** for multi-object tracking
- **PyTorch** for deep learning
