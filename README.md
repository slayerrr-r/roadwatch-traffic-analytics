# RoadWatch — Video-Based Traffic Analytics

Intermediate computer-vision project using pretrained YOLO, object tracking, OpenCV, and Streamlit to analyze road videos.

## Features
- Detect objects in uploaded road videos with pretrained YOLO weights.
- Track objects across frames using Ultralytics tracking (ByteTrack configuration).
- Approximate vehicle counts when tracked objects cross a configurable virtual line.
- Download annotated MP4, frame-level detection CSV, crossing-events CSV, and traffic-summary CSV.
- Confidence/IoU controls, frame limits, reusable inference functions, and unit tests.

**Honest scope:** the model is pretrained, not custom-trained. Line crossing is approximate, track IDs can switch, and results are not a calibrated traffic measurement. Speed estimation is intentionally excluded because it requires camera calibration. Verify on your own footage before reporting results.

## Stack
Python 3.10/3.11, Ultralytics YOLO, OpenCV, Streamlit, Pandas, Pytest.

## Setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
If activation is blocked, run ` .\.venv\Scripts\python.exe -m pip install -r requirements.txt` and then `pip install -e .` with the same executable.

## Run
```bash
streamlit run app.py
```
Upload a short road video (`mp4`, `mov`, `avi`, `mkv`), set a virtual counting line in pixel coordinates, and click **Analyze video**. First run may download model weights. Start with a 10–30 second clip.

## Tests
```bash
pytest
```
The unit tests cover line-crossing and aggregation logic and do not require model weights.

## Pipeline
1. OpenCV reads frames.
2. YOLO predicts bounding boxes/classes; tracker attempts to associate detections over time.
3. Bottom-center of each box approximates the ground-contact point.
4. A track is counted once when that point crosses the virtual line.
5. Annotated video and CSV reports are generated.

## Interview concepts
- Detection vs. tracking; confidence thresholds; IoU and non-maximum suppression; temporary track IDs; line-crossing geometry; occlusion and camera-angle failure cases.

## Resume wording (only after running and validating)
**RoadWatch — Video-Based Traffic Analytics | Python, YOLO, OpenCV, Streamlit**
- Built an interactive road-video analytics app using pretrained YOLO detection and object tracking.
- Implemented virtual-line crossing counts by class, annotated video output, and CSV exports for detections and traffic summaries.

Only list features you have personally run and verified. Project code is MIT licensed; Ultralytics and model weights have separate terms—review them before commercial use.