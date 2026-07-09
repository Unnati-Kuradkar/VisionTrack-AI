import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from rfdetr import RFDETRNano
import supervision as sv
import av
import cv2
from streamlit_autorefresh import st_autorefresh
from collections import Counter

if "object_counts" not in st.session_state:
    st.session_state.object_counts = {}

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

#st_autorefresh(interval=1000, key="refresh")
# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="VisionTrack AI",
    layout="wide"
)


st.markdown("""
<style>
.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="fade-in">

<h1 style='text-align:center;'>
🎥 VisionTrack AI
</h1>

<p style='text-align:center;color:gray;'>
Real-Time Object Detection & Tracking Dashboard
</p>

<div class="live-status">
🟢 LIVE DETECTION ACTIVE
</div>

</div>
""", unsafe_allow_html=True)

# --------------------------------
# GLOBAL COUNTS
# --------------------------------

if "object_counts" not in st.session_state:
    st.session_state.object_counts = {}

# --------------------------------
# LOAD MODEL
# --------------------------------

@st.cache_resource
def load_model():
    return RFDETRNano()

model = load_model()


# --------------------------------
# ANNOTATORS
# --------------------------------

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# --------------------------------
# VIDEO PROCESSOR
# --------------------------------
latest_counts = {}
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.last_detections = None
        self.counts = {}


    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Faster processing
        img_small = cv2.resize(img, (280, 210))

        self.frame_count += 1

        # Run RF-DETR every 10th frame
        if self.frame_count % 10 == 0:
            self.last_detections = model.predict(img_small)

        detections = self.last_detections

        if detections is None:
            return av.VideoFrame.from_ndarray(
                img_small,
                format="bgr24"
            )
        


        # ----------------------------
        # CATEGORY COUNTS
        # ----------------------------

        counts = {}
     

        for class_name in detections.data["class_name"]:
            counts[class_name] = counts.get(class_name, 0) + 1

        self.counts = counts
        try:
            st.session_state.object_counts = counts
        except:
            pass
        print("SELF COUNTS =", self.counts)
        print("DETECTED COUNTS:", counts)
        # Draw Boxes
        annotated = box_annotator.annotate(
            scene=img_small.copy(),
            detections=detections
        )

        # Draw Labels
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections
        )

        label_text = " | ".join(
            [f"{k}: {v}" for k, v in counts.items()]
        )

        cv2.putText(
            annotated,
            label_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# --------------------------------
# CAMERA + INFO PANEL
# --------------------------------

col1, col2 = st.columns([0.8,1.2])

with col1:

    st.markdown(
        "<h2>📹 Live Camera Feed</h2>",
        unsafe_allow_html=True
    )

    ctx = webrtc_streamer(
        key="visiontrack",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=False
    )

with col2:

    st.markdown("""
    <div class="info-card">

    <h2>📊 Detection Information</h2>

    • RF-DETR Nano Model<br>
    • Real-Time Object Detection<br>
    • Live Webcam Processing<br>

    </div>
    """, unsafe_allow_html=True)

   