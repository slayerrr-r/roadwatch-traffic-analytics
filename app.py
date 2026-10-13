"""Streamlit interface for RoadWatch."""
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from roadwatch.pipeline import analyze_video, events_to_dataframe, load_model, rows_to_dataframe

st.set_page_config(page_title="RoadWatch Traffic Analytics", page_icon="🚦", layout="wide")
st.title("🚦 RoadWatch — Video-Based Traffic Analytics")
st.write("Detect, track, and count objects crossing a virtual line in road footage.")
st.caption("Uses pretrained YOLO weights and approximate line-crossing analytics. Track IDs and counts can be imperfect in crowded or occluded scenes.")
with st.sidebar:
    st.header("Model settings")
    model_name = st.selectbox("Pretrained model", ["yolov8n.pt", "yolov8s.pt"])
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
    max_frames = st.number_input("Maximum frames (0 = entire video)", min_value=0, max_value=100000, value=900, step=100)
    st.info("First run downloads model weights. Start with a short clip.")
@st.cache_resource(show_spinner="Loading YOLO model...")
def cached_model(name): return load_model(name)
uploaded = st.file_uploader("Upload a road video", type=["mp4","mov","avi","mkv"])
if uploaded:
    st.video(uploaded)
    st.subheader("Virtual counting line (pixel coordinates)")
    st.caption("Defaults are a starting point for a 640×360 video. Adjust endpoints to your video's resolution and road layout.")
    a,b,c,d = st.columns(4)
    x1=a.number_input("Start X",min_value=0,value=50,step=10)
    y1=b.number_input("Start Y",min_value=0,value=180,step=10)
    x2=c.number_input("End X",min_value=1,value=590,step=10)
    y2=d.number_input("End Y",min_value=0,value=180,step=10)
    if st.button("Analyze video", type="primary"):
        if x1==x2 and y1==y2: st.error("Choose two different line endpoints.")
        else:
            try:
                with tempfile.TemporaryDirectory() as td:
                    td=Path(td); suffix=Path(uploaded.name).suffix or ".mp4"
                    source=td/f"input{suffix}"; output=td/"roadwatch_annotated.mp4"
                    source.write_bytes(uploaded.getvalue()); model=cached_model(model_name)
                    with st.spinner("Detecting, tracking, and analyzing frames..."):
                        detections, events, report=analyze_video(source,output,model,(int(x1),int(y1)),(int(x2),int(y2)),conf=float(conf),iou=float(iou),max_frames=int(max_frames) if max_frames>0 else None)
                    video_bytes=output.read_bytes()
                stats=report["stats"]
                cols=st.columns(4)
                cols[0].metric("Frames",stats["frames_processed"]); cols[1].metric("Unique tracks",stats["unique_tracks_seen"]); cols[2].metric("Crossings",stats["crossing_events"]); cols[3].metric("Duration processed",f'{stats["duration_processed_seconds"]:.1f}s')
                st.subheader("Annotated output"); st.video(video_bytes)
                st.download_button("Download annotated MP4",video_bytes,"roadwatch_annotated.mp4","video/mp4")
                summary=pd.DataFrame(report["summary"],columns=["class_name","direction","count"])
                st.subheader("Traffic summary")
                if summary.empty: st.info("No tracked object crossed the line. Adjust the line or try another clip.")
                else:
                    st.dataframe(summary,use_container_width=True,hide_index=True)
                    st.bar_chart(summary.groupby("class_name")["count"].sum())
                st.download_button("Download traffic summary CSV",summary.to_csv(index=False).encode(),"traffic_summary.csv","text/csv")
                det_df=rows_to_dataframe(detections); st.subheader("Frame-level detections")
                st.dataframe(det_df.head(500),use_container_width=True,hide_index=True); st.caption(f"Showing up to 500 of {len(det_df)} rows.")
                st.download_button("Download detections CSV",det_df.to_csv(index=False).encode(),"detections.csv","text/csv")
                event_df=events_to_dataframe(events); st.subheader("Crossing events")
                st.dataframe(event_df,use_container_width=True,hide_index=True)
                st.download_button("Download crossing events CSV",event_df.to_csv(index=False).encode(),"crossing_events.csv","text/csv")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}"); st.caption("Check the video, dependencies, line coordinates, and model download.")
else: st.info("Upload a short road video to begin.")
