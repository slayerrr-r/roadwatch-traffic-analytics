"""YOLO tracking and video analytics pipeline."""
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO
from roadwatch.analytics import crossed_line, summarize_crossings

def load_model(model_name="yolov8n.pt"):
    """Load pretrained model; weights download on first use if absent."""
    return YOLO(model_name)

def analyze_video(input_path, output_path, model, line_start, line_end,
                  conf=0.25, iou=0.45, max_frames=None, tracker="bytetrack.yaml"):
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened(): raise ValueError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0: fps = 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release(); raise ValueError("Video has invalid frame dimensions.")
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if not writer.isOpened():
        cap.release(); raise RuntimeError("Could not create output MP4.")
    rows, events, last_pos, counted = [], [], {}, set()
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            result = model.track(source=frame, persist=True, tracker=tracker, conf=conf, iou=iou, verbose=False)[0]
            annotated = result.plot()
            cv2.line(annotated, line_start, line_end, (255,255,0), 3)
            cv2.putText(annotated, "COUNTING LINE", (line_start[0], max(25,line_start[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                ids = getattr(boxes, "id", None)
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls.item()); cls_name = result.names.get(cls_id, str(cls_id))
                    score = float(box.conf.item()); x1,y1,x2,y2 = [float(v) for v in box.xyxy[0].tolist()]
                    track_id = int(ids[i].item()) if ids is not None else None
                    rows.append({"frame":frame_index,"timestamp_seconds":round(frame_index/fps,3),"track_id":track_id,"class_id":cls_id,"class_name":cls_name,"confidence":round(score,5),"x1":round(x1,2),"y1":round(y1,2),"x2":round(x2,2),"y2":round(y2,2)})
                    if track_id is None: continue
                    point = ((x1+x2)/2.0, y2)  # approximate ground-contact point
                    previous = last_pos.get(track_id)
                    if previous is not None and track_id not in counted:
                        direction = crossed_line(previous, point, line_start, line_end)
                        if direction:
                            events.append({"track_id":track_id,"class_name":cls_name,"direction":direction,"frame":frame_index,"timestamp_seconds":round(frame_index/fps,3)})
                            counted.add(track_id)
                    last_pos[track_id] = point
            writer.write(annotated); frame_index += 1
            if max_frames is not None and frame_index >= max_frames: break
    finally:
        cap.release(); writer.release()
    if frame_index == 0: raise ValueError("No readable frames were found in the video.")
    stats = {"frames_processed":frame_index,"source_fps":fps,"duration_processed_seconds":round(frame_index/fps,3),"unique_tracks_seen":len(last_pos),"tracks_counted":len(counted),"crossing_events":len(events)}
    return rows, events, {"summary":summarize_crossings(events),"stats":stats}

def rows_to_dataframe(rows):
    return pd.DataFrame(rows, columns=["frame","timestamp_seconds","track_id","class_id","class_name","confidence","x1","y1","x2","y2"])

def events_to_dataframe(events):
    return pd.DataFrame(events, columns=["track_id","class_name","direction","frame","timestamp_seconds"])
