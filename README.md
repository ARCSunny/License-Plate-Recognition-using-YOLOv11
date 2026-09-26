# License Plate Detection and OCR

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

```text
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

## Requirements

Python 3.9+ is recommended.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

For Google Colab, the notebook also installs the main computer-vision packages directly.

## Dataset

The notebook downloads the license-plate dataset from Roboflow using the Roboflow Python SDK.

The dataset configuration is adjusted to use the following YOLO directory structure:

```text
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

```python
from getpass import getpass

api_key = getpass("API Key: ")
```

Do **not** commit your Roboflow API key or other credentials to GitHub.

## Model Training

The project uses the YOLO11 small model:

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
```

Training is configured with:

```python
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

```python
best = "/content/runs/license_plate/weights/best.pt"

model = YOLO(best)
metrics = model.val(data=str(yaml_path), device=0)

print(f"mAP@50: {metrics.box.map50:.3f}")
print(f"mAP@50-95: {metrics.box.map:.3f}")
```

The training results are also saved as:

```text
/content/runs/license_plate/results.png
```

## OCR

After license plates are detected, the detected plate regions are passed to EasyOCR.

```python
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

```text
Original OCR result: "DHA-1234"
Processed result:    "DHA1234"
```

## Video Tracking

The trained detector is used with **ByteTrack** to maintain object identities across video frames:

```python
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

```python
OCR_EVERY_N_FRAMES = 3
```

The OCR results for each tracking ID are stored and the most frequently recognized plate text is displayed for that tracked object.

## Running the Notebook

### Option 1: Google Colab

1. Open `LicensePlateDetection.ipynb` in Google Colab.
2. Select a GPU runtime:
   - **Runtime → Change runtime type → GPU**
3. Run the setup cells.
4. Provide your Roboflow API key when prompted.
5. Download and prepare the dataset.
6. Train or load the YOLO model.
7. Evaluate the model.
8. Upload a vehicle video when prompted.
9. Run the tracking and OCR pipeline.
10. Download the generated video and model weights.

### Option 2: Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
LicensePlateDetection.ipynb
```

> Some notebook cells use Google Colab-specific functionality such as `google.colab.files`. For local execution, those upload/download cells may need to be replaced with standard local file paths.

## Using the Included Model

The repository includes the trained model:

```text
best.pt
```

Load it with Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

You can then run detection on an image or video:

```python
results = model.predict(
    source="input.jpg",
    conf=0.35
)
```

For tracking:

```python
results = model.track(
    source="input.mp4",
    tracker="bytetrack.yaml",
    conf=0.35,
    persist=True
)
```

## Output

The notebook generates an annotated video containing:

- Bounding boxes around detected license plates
- Detection confidence
- Recognized license plate text
- Tracking IDs internally used to maintain plate identity

The final re-encoded video is:

```text
/content/output/result.mp4
```

The trained model is saved as:

```text
/content/runs/license_plate/weights/best.pt
```

## Memory and GPU Monitoring

The notebook includes helper functions for monitoring system RAM and GPU memory:

```python
mem_check("startup")
```

and:

```python
free_mem()
```

These functions can be useful when running training, OCR, and video processing on limited hardware.

## Configuration

Several settings can be adjusted directly in the notebook.

### Detection Confidence

```python
conf=0.35
```

Increase the value to reduce low-confidence detections, or decrease it to detect more potential plates.

### IoU Threshold

```python
iou=0.5
```

This controls the intersection-over-union threshold used during tracking/detection.

### Image Size

```python
imgsz=640
```

Larger images can potentially improve detection of small plates but require more GPU memory.

### OCR Frequency

```python
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

## Privacy and Responsible Use

License plate information can be sensitive personal or vehicle-related data depending on the jurisdiction and use case.

When using this project:

- Process video only when you have appropriate authorization.
- Follow applicable privacy and data-protection laws.
- Avoid publishing identifiable vehicle information unnecessarily.
- Secure stored videos, images, OCR results, and model outputs.
- Do not commit private datasets, credentials, or sensitive recordings to the repository.

## License

See the repository's `License` file or the license associated with the underlying model, dataset, and dependencies before redistributing or deploying this project.

## Acknowledgements

This project uses:

- **Ultralytics YOLO** for object detection
- **Roboflow** for dataset management
- **EasyOCR** for optical character recognition
- **OpenCV** for computer vision and video processing
- **ByteTrack** for multi-object tracking
- **PyTorch** for deep learning

## Author

**Md. Ashiqur Rahman Chowdhury Sunny**

---

If you use or modify this project, consider documenting your dataset version, training configuration, evaluation metrics, and hardware so that results can be reproduced.
