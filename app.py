import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import tempfile
import time
from collections import deque
import pickle
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import plotly.express as px
import os
from twilio.rest import Client
import av

#imports to improve system UI here
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="ActionNet System", layout="wide", initial_sidebar_state="expanded")

#*******UI improvements for better user experience and aesthetics********
st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

.main {
    background-color:#0B1120;
}

.block-container{
    padding-top:1rem;
}

[data-testid="stMetric"]{
    background:#111827;
    border:1px solid #1F2937;
    padding:20px;
    border-radius:15px;
}

h1{
    font-weight:800;
}

.hero-box{
    background:linear-gradient(135deg,#2563EB,#7C3AED);
    padding:2rem;
    border-radius:20px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# --- 2. LOAD RANDOM FOREST MODEL ---
@st.cache_resource
def load_ml_model():
    with open('Human_actionV1.3.pkl', 'rb') as f:
        return pickle.load(f)

model = load_ml_model()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- 3. SESSION STATE (Memory across pages) ---
WINDOW_SIZE = 15
if "pose_buffer" not in st.session_state:
    st.session_state.pose_buffer = deque(maxlen=WINDOW_SIZE)
if "custom_buffer" not in st.session_state:
    st.session_state.custom_buffer = deque(maxlen=WINDOW_SIZE)
if "history" not in st.session_state:
    st.session_state.history = []
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.55

# --- 4. CORE INFERENCE ENGINE (410 FEATURES) ---
def process_frame(frame, source_name):
    """Shared pipeline for both Video Upload and Live WebRTC."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1) as pose_model:
        results = pose_model.process(image_rgb)
        
        prediction = "Analyzing..."
        confidence = 0.0
        
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            
            # A. Base Spatial Features
            spatial_features = [val for lm in pose for val in (lm.x, lm.y, lm.z, lm.visibility)]
            st.session_state.pose_buffer.append(spatial_features)
            
            # # B. Custom Geometry Extraction
            # l_shoulder = np.array([pose[11].x, pose[11].y])
            # r_shoulder = np.array([pose[12].x, pose[12].y])
            # l_wrist = np.array([pose[15].x, pose[15].y])
            # r_wrist = np.array([pose[16].x, pose[16].y])
            # l_hip = np.array([pose[23].x, pose[23].y])
            # r_hip = np.array([pose[24].x, pose[24].y])
            # l_ankle = np.array([pose[27].x, pose[27].y])
            # r_ankle = np.array([pose[28].x, pose[28].y])
            
            # shoulder_width = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
            
            # wrist_dist = np.linalg.norm(l_wrist - r_wrist) / shoulder_width
            # ankle_dist = np.linalg.norm(l_ankle - r_ankle) / shoulder_width
            # l_hand_raised = (pose[15].y - pose[11].y) / shoulder_width
            # r_hand_raised = (pose[16].y - pose[12].y) / shoulder_width
            # torso_height = np.linalg.norm(l_shoulder - l_hip) / shoulder_width
            
            # custom_frame_features = [
            #     wrist_dist, ankle_dist, l_hand_raised, r_hand_raised, 
            #     torso_height, pose[23].y, pose[24].y 
            # ]
            # --- HELPER ARRAYS FOR EASY MATH --- modified
            l_shoulder = np.array([pose[11].x, pose[11].y])
            r_shoulder = np.array([pose[12].x, pose[12].y])
            l_wrist = np.array([pose[15].x, pose[15].y])
            r_wrist = np.array([pose[16].x, pose[16].y])
            l_hip = np.array([pose[23].x, pose[23].y])
            r_hip = np.array([pose[24].x, pose[24].y])
            l_ankle = np.array([pose[27].x, pose[27].y])
            r_ankle = np.array([pose[28].x, pose[28].y])
            
            # --- CALCULATE REFERENCE SCALE ---
            shoulder_width = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
            
            # 1. Define centralized points for the core body axis
            mid_shoulder = (l_shoulder + r_shoulder) / 2.0
            mid_hip = (l_hip + r_hip) / 2.0
            
            # 2. TORSO ANGLE (Calculated on this single frame)
            # np.arctan2(y, x) -> using the numpy arrays we just created
            torso_angle = np.arctan2(mid_hip[1] - mid_shoulder[1], mid_hip[0] - mid_shoulder[0])
            
            # 3. ENGINEERED FEATURES (Appended per frame)
            wrist_dist = np.linalg.norm(l_wrist - r_wrist) / shoulder_width
            ankle_dist = np.linalg.norm(l_ankle - r_ankle) / shoulder_width
            l_hand_raised = (pose[15].y - pose[11].y) / shoulder_width
            r_hand_raised = (pose[16].y - pose[12].y) / shoulder_width
            torso_height = np.linalg.norm(mid_shoulder - mid_hip) / shoulder_width
            
            # 4. Assemble the 7 custom features EXACTLY matching training order
            custom_frame_features = [
                wrist_dist, 
                ankle_dist, 
                l_hand_raised, 
                r_hand_raised, 
                torso_height, 
                torso_angle, 
                mid_hip[1] # mid_hip_y
            ]
            
            st.session_state.custom_buffer.append(custom_frame_features)
            
            # C. Trigger Prediction when buffer is full
            if len(st.session_state.pose_buffer) == WINDOW_SIZE:
                try:
                    # Standard Stats
                    buffer_array = np.array(st.session_state.pose_buffer)
                    temporal_mean = np.mean(buffer_array, axis=0)
                    temporal_std = np.std(buffer_array, axis=0)
                    temporal_range = np.ptp(buffer_array, axis=0)
                    
                    # Custom Stats
                    custom_array = np.array(st.session_state.custom_buffer)
                    custom_means = np.mean(custom_array, axis=0)
                    custom_variances = np.var(custom_array, axis=0)
                    
                    # Combine to 410 features
                    combined_features = np.concatenate([
                        temporal_mean, temporal_std, temporal_range, 
                        custom_means, custom_variances
                    ])
                    
                    X = pd.DataFrame([combined_features])
                    
                    # Random Forest Inference
                    raw_confidence = np.max(model.predict_proba(X))
                    if raw_confidence >= st.session_state.confidence_threshold:
                        prediction = model.predict(X)[0]
                        confidence = raw_confidence * 100
                        
                        # Log to History (throttle to prevent spam)
                        if len(st.session_state.history) == 0 or st.session_state.history[-1]['action'] != prediction:
                            st.session_state.history.append({
                                "time": time.strftime("%H:%M:%S"),
                                "source": source_name,
                                "action": prediction,
                                "confidence": confidence
                            })
                    else:
                        prediction = "Unknown (Low Conf)"
                        confidence = raw_confidence * 100
                        
                except Exception as e:
                    prediction = "Shape Error"
                    
            # Draw Skeleton
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        return frame, prediction, confidence

# --- 5. UI HELPER FUNCTION ---
def render_metrics(action, conf, fps):
    cols = st.columns(3)
    cols[0].metric("Detected Action", action)
    cols[1].metric("Confidence", f"{conf:.1f}%")
    cols[2].metric("Processing FPS", f"{fps:.1f}")

# --- 5.5 WEBCAM NETWORK CONFIGURATION ---
# @st.cache_data
# def get_ice_servers():
#     """Fetches TURN servers from Twilio, falls back to Google STUN if keys are missing."""
#     if not account_sid or not auth_token:
#         try:
#             # Checks Streamlit Secrets first
#             account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
#             auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
#             client = Client(account_sid, auth_token)
#             token = client.tokens.create()
#             return token.ice_servers
#         except Exception as e:
#             # Fallback for local testing or missing secrets
#             print(f"Twilio Secrets missing or failed. Using fallback Google STUN. Error: {e}")
#             return [{"urls": ["stun:stun.l.google.com:19302"]}]
#     # Fetch the TURN servers from Twilio
#     client = Client(account_sid, auth_token)
#     token = client.tokens.create()
#     return token.ice_servers

@st.cache_data(ttl=43200) # Force a fresh token fetch every 12 hours
def get_ice_servers():
    """Fetches TURN servers from Twilio, falls back to Google STUN if keys are missing."""
    try:
        # 1. Fetch credentials directly from Streamlit Secrets
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        
        # 2. Request a fresh token from Twilio
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        
        return token.ice_servers
        
    except Exception as e:
        # Fallback for local testing or if Twilio API fails
        print(f"Twilio API failed or secrets missing. Using fallback STUN. Error: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# --- 6. SIDEBAR ROUTING ---
st.sidebar.title("ActionNet")

#******Upgrading to a more modern and visually appealing sidebar navigation using streamlit_option_menu*
# page = st.sidebar.radio("Navigation", ["About", "Upload Video", "Live Stream", "History", "Settings"])

with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        width=100
    )

    page = option_menu(
        "ActionNet",
        [
            
            "Upload Video",
            "Live Stream",
            "History",
            "About",
            "Settings"
        ],
        icons=[
            
            "upload",
            "camera-video",
            "graph-up",
            "speedometer2",
            "gear"
        ],
        default_index=0
    )

# if page == "About":

    

# ==========================================
# PAGE 1: ABOUT
# ==========================================
# if page == "About":
#     st.title("About ActionNet")
#     st.markdown("ActionNet is a real-time Human Action Recognition system powered by MediaPipe and an optimized Random Forest classifier.")
#     st.markdown("It processes a **15-frame rolling window**, analyzing 410 temporal and geometric features—such as shoulder-normalized wrist distance and hip variance—to detect actions like Walking, Jumping, and Clapping with high precision.")

# ==========================================
# PAGE 1: ABOUT
# ==========================================
if page == "About":
    # st.title("ℹ️ About ActionNet")
    # st.markdown("ActionNet is a real-time Human Action Recognition (HAR) system built for edge inference. It transforms raw video streams into a structured temporal dataset, utilizing a **15-frame rolling window** to analyze human motion.")
    st.markdown("""
    <div class='hero-box'>
    <h1>ActionNet AI</h1>
    <h4>Human Action Recognition System</h4>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Model",
        "RF v1.1"
    )

    c2.metric(
        "Features",
        "410"
    )

    c3.metric(
        "Window",
        "15 Frames"
    )

    c4.metric(
        "Status",
        "Online"
    )

    style_metric_cards(background_color="#1F2937", border_color="#4B5563", border_size_px=2, border_radius_px=15)

    st.divider()

    col1,col2 = st.columns([2,1])

    with col1:

        st.subheader("1. System Overview")
        st.markdown("ActionNet captures video input and uses **MediaPipe Pose** to extract 33 3D landmarks per frame. It computes scale-normalized geometric features (e.g., wrist distance normalized by shoulder width) and tracks their temporal variance across a 15-frame buffer. This structured dataset is fed into an optimized Random Forest classifier that detects actions with high accuracy, even in noisy conditions.")
        st.image("https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png", 
                 caption="MediaPipe Pose Landmarks (33 Keypoints)", 
                 width = "stretch")

    with col2:

        st.info("""
        ### Platform Features
        
         Live Detection
                
        
         Video Upload
                
        
         MediaPipe Tracking
                
        
         Random Forest Classification
                
        
         Analytics Dashboard
        """, width="stretch")
    # st.divider()
    
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.subheader("1. Spatial & Geometric Extraction")
    #     st.markdown("The system uses **MediaPipe Pose** to extract 33 3D anatomical landmarks per frame. Instead of relying purely on raw coordinates, ActionNet calculates scale-normalized geometric ratios (e.g., wrist-to-wrist distance normalized by shoulder width) to ensure the model performs accurately regardless of the subject's distance from the camera.")
        
        # st.markdown("**Euclidean Distance Formula (Scale Normalization):**")
        # st.latex(r"d(p_1, p_2) = \frac{\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}}{W_{shoulder}}")
        
        # st.markdown("**Temporal Variance Formula (Motion Tracking):**")
        # st.markdown("To detect dynamic actions like jumping or walking, the variance of these features is calculated across the 15-frame buffer:")
        # st.latex(r"\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2")

    # with col2:
        # Using the official MediaPipe topology image URL so it loads automatically
        
    # st.divider()

    # st.subheader("2. The Random Forest Classifier")
    # st.markdown("The extracted sequence is flattened into a **410-feature tabular array** containing the Mean, Standard Deviation, and Peak-to-Peak Range of the movements. This array is passed into an optimized Random Forest algorithm.")
    # st.markdown("Random Forest is an ensemble learning method that constructs a multitude of decision trees. It splits the data based on the features that minimize **Gini Impurity**, making it incredibly resilient to the overlapping noise of human motion.")
    
    # st.markdown("**Gini Impurity Formula:**")
    # st.latex(r"Gini = 1 - \sum_{i=1}^{C} (p_i)^2")

# ==========================================
# PAGE 2: UPLOAD VIDEO
# ==========================================
elif page == "Upload Video":
    # st.title("📼 Video Upload Inference") ---upgrading the UI
    st.markdown("""
<div class='hero-box'>
<h1>📼 Video Analysis</h1>
<p>Upload a video and receive action predictions in real time.</p>
</div>
""", unsafe_allow_html=True)
    
    st.write("")
    #still part of the UI upgrade, making the upload button more visually appealing
    m1,m2,m3 = st.columns(3)

    m1.metric("Supported Formats","MP4 AVI MOV")
    m2.metric("Inference Engine","Random Forest")
    m3.metric("Pose Tracker","MediaPipe")

    style_metric_cards(background_color="#1F2937", border_color="#4B5563", border_size_px=2, border_radius_px=15)
    uploaded_file = st.file_uploader("Upload an MP4 or AVI file", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            video_placeholder = st.empty()
            metrics_placeholder = st.empty()
        with col2:
            st.markdown("### Metadata")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write("**Status:** Processing")
            
        # OpenCV needs a physical file path, so we write the upload to a temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        start_time = time.time()
        frame_count = 0
        
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret: break
            
        #     frame_count += 1
        #     frame = cv2.resize(frame, (640, 480))
            
        #     # Pass to our main engine
        #     processed_frame, action, conf = process_frame(frame, "Upload")
            
        #     # UI Overlay
        #     cv2.rectangle(processed_frame, (10, 10), (350, 90), (0, 0, 0), -1)
        #     cv2.putText(processed_frame, f"ACTION: {action}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        #     cv2.putText(processed_frame, f"CONF: {conf:.1f}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        #     video_placeholder.image(processed_frame, channels="BGR", width = "stretch")
            
        #     if frame_count % 5 == 0:
        #         with metrics_placeholder.container():
        #             fps = frame_count / (time.time() - start_time)
        #             render_metrics(action, conf, fps)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            
            # --- FIX: Resize while maintaining aspect ratio ---
            h, w = frame.shape[:2]
            target_width = 640
            aspect_ratio = target_width / float(w)
            target_height = int(h * aspect_ratio)
            
            frame = cv2.resize(frame, (target_width, target_height))
            # --------------------------------------------------
            
            # Pass to our main engine
            processed_frame, action, conf = process_frame(frame, "Upload")
            
            # UI Overlay
            cv2.rectangle(processed_frame, (10, 10), (350, 90), (0, 0, 0), -1)
            cv2.putText(processed_frame, f"ACTION: {action}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"CONF: {conf:.1f}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            video_placeholder.image(processed_frame, channels="BGR", width = "stretch")
            
            if frame_count % 5 == 0:
                with metrics_placeholder.container():
                    fps = frame_count / (time.time() - start_time)
                    render_metrics(action, conf, fps)
                    
        cap.release()
        st.success("Video processing complete!")

        # Delete the physical file from the Hugging Face server so it doesn't crash!
        try:
            os.remove(tfile.name)
        except Exception as e:
            pass
            
        st.success("Video processing complete! Temporary server files cleaned up.")

# ==========================================
# PAGE 3: LIVE STREAM
# ==========================================
# elif page == "Live Stream":
#     st.title("🎥 Real-Time WebRTC Inference")
    
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         video_placeholder = st.empty()
        
#         def video_callback(frame):
#             img = frame.to_ndarray(format="bgr24")
#             processed_img, action, conf = process_frame(img, "Live Stream")
            
#             cv2.rectangle(processed_img, (10, 10), (350, 90), (0, 0, 0), -1)
#             cv2.putText(processed_img, f"LIVE: {action}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             cv2.putText(processed_img, f"CONF: {conf:.1f}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
#             return mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_img).to_msg_frame()

#         ctx = webrtc_streamer(
#             key="live-stream",
#             mode=WebRtcMode.SENDRECV,
#             rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
#             video_frame_callback=video_callback,
#             media_stream_constraints={"video": True, "audio": False},
#             async_processing=True,
#         )

# ==========================================
# PAGE 3: LIVE STREAM
# ==========================================
elif page == "Live Stream":
    # st.title("🎥 Real-Time WebRTC Inference") --- apply the UI upgrade here too
    st.markdown("""
<div class='hero-box'>
<h1>🎥 Live Action Monitoring</h1>
<p>Monitor actions in real time using webcam inference.</p>
</div>
""", unsafe_allow_html=True)
    
    
    st.write("")
    def confidence_gauge(conf):

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf,
            title={"text":"Confidence"},
            gauge={
                "axis":{"range":[0,100]}
            }
        ))

        st.plotly_chart(
            fig,
            width = "stretch"
        )
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Initializing secure WebRTC stream...")
                
        # 1. Create a Thread-Safe Class for the Video Stream
        class ActionProcessor:
            def __init__(self):
                # Buffers live safely inside the instance, not session_state
                self.pose_buffer = deque(maxlen=WINDOW_SIZE)
                self.custom_buffer = deque(maxlen=WINDOW_SIZE)
                
                # MASSIVE PERFORMANCE BOOST: Initialize MediaPipe ONCE here
                # instead of opening and closing it on every single frame.
                self.pose_model = mp_pose.Pose(
                    min_detection_confidence=0.7, 
                    min_tracking_confidence=0.7, 
                    model_complexity=1
                )
                
                # Default states
                self.action = "Buffering..."
                self.confidence = 0.0
                self.threshold = 0.65

            def recv(self, frame):
                # Convert the incoming WebRTC frame
                img = frame.to_ndarray(format="bgr24")
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Process with the pre-loaded MediaPipe model
                results = self.pose_model.process(image_rgb)
                
                if results.pose_landmarks:
                    pose = results.pose_landmarks.landmark
                    
                    # A. Base Spatial Features
                    spatial_features = [val for lm in pose for val in (lm.x, lm.y, lm.z, lm.visibility)]
                    self.pose_buffer.append(spatial_features)
                    
                    # B. Custom Geometry Extraction
                    l_shoulder = np.array([pose[11].x, pose[11].y])
                    r_shoulder = np.array([pose[12].x, pose[12].y])
                    l_wrist = np.array([pose[15].x, pose[15].y])
                    r_wrist = np.array([pose[16].x, pose[16].y])
                    l_hip = np.array([pose[23].x, pose[23].y])
                    r_hip = np.array([pose[24].x, pose[24].y])
                    l_ankle = np.array([pose[27].x, pose[27].y])
                    r_ankle = np.array([pose[28].x, pose[28].y])
                    
                    shoulder_width = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
                    
                    wrist_dist = np.linalg.norm(l_wrist - r_wrist) / shoulder_width
                    ankle_dist = np.linalg.norm(l_ankle - r_ankle) / shoulder_width
                    l_hand_raised = (pose[15].y - pose[11].y) / shoulder_width
                    r_hand_raised = (pose[16].y - pose[12].y) / shoulder_width
                    torso_height = np.linalg.norm(l_shoulder - l_hip) / shoulder_width
                    
                    custom_frame_features = [
                        wrist_dist, ankle_dist, l_hand_raised, r_hand_raised, 
                        torso_height, pose[23].y, pose[24].y 
                    ]
                    self.custom_buffer.append(custom_frame_features)
                    
                    # C. Trigger Prediction when buffer is full
                    if len(self.pose_buffer) == WINDOW_SIZE:
                        try:
                            # Standard Stats
                            buffer_array = np.array(self.pose_buffer)
                            temporal_mean = np.mean(buffer_array, axis=0)
                            temporal_std = np.std(buffer_array, axis=0)
                            temporal_range = np.ptp(buffer_array, axis=0)
                            
                            # Custom Stats
                            custom_array = np.array(self.custom_buffer)
                            custom_means = np.mean(custom_array, axis=0)
                            custom_variances = np.var(custom_array, axis=0)
                            
                            # Combine to 410 features
                            combined_features = np.concatenate([
                                temporal_mean, temporal_std, temporal_range, 
                                custom_means, custom_variances
                            ])
                            
                            X = pd.DataFrame([combined_features])
                            
                            # Random Forest Inference
                            raw_confidence = np.max(model.predict_proba(X))
                            
                            # Use the safely passed threshold variable
                            if raw_confidence >= self.threshold:
                                self.action = model.predict(X)[0]
                                self.confidence = raw_confidence * 100
                            else:
                                self.action = "Unknown (Low Conf)"
                                self.confidence = raw_confidence * 100
                                
                        except Exception as e:
                            self.action = "Shape Error"
                            
                    # Draw Skeleton
                    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                # HUD Overlay
                cv2.rectangle(img, (10, 10), (350, 90), (0, 0, 0), -1)
                cv2.putText(img, f"LIVE: {self.action}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(img, f"CONF: {self.confidence:.1f}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Return the processed frame to the WebRTC stream
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        # 2. Mount the Streamer using the custom class
        ctx = webrtc_streamer(
            key="live-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
            video_processor_factory=ActionProcessor, # the Upgrade
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # 3. Thread-Safe Variable Passing
        # If the user changes the slider on the Settings page, this passes the 
        # new value across the boundary into the running background thread.
        if ctx.video_processor:
            ctx.video_processor.threshold = st.session_state.confidence_threshold

# ==========================================
# PAGE 4: HISTORY
# ==========================================
# elif page == "History":
#     st.title("📜 Action History Log")
#     if len(st.session_state.history) > 0:
#         df = pd.DataFrame(st.session_state.history)
#         st.dataframe(df, width = "stretch")
        
#         fig = px.line(df, x="time", y="confidence", color="action", title="Confidence Timeline")
#         st.plotly_chart(fig, width = "stretch")
        
#         if st.button("Clear History"):
#             st.session_state.history = []
#             st.rerun()
#     else:
#         st.info("No actions logged yet.")

# ==========================================
# PAGE 4: HISTORY
# ==========================================
elif page == "History":
    # st.title("📜 Action History & Analytics") --- apply upgrade to the UI
    st.markdown("""
<div class='hero-box'>
<h1>📊 Analytics Center</h1>
<p>Review detections and model performance.</p>
</div>
""", unsafe_allow_html=True)
    
    if len(st.session_state.history) > 0:
        # Convert history dictionary to Pandas DataFrame
        df = pd.DataFrame(st.session_state.history)
        
        # --- NEW ANALYTICS SECTION: Average Confidence Per Class ---
        st.subheader("📊 Performance Analytics")
        
        # Group by action and calculate the mean confidence
        avg_conf_df = df.groupby('action')['confidence'].mean().reset_index()
        avg_conf_df = avg_conf_df.rename(columns={'confidence': 'Average Confidence (%)'})
        
        # Create a Bar Chart for the averages
        fig_bar = px.bar(
            avg_conf_df, 
            x='action', 
            y='Average Confidence (%)',
            color='action',
            title="Average Model Confidence by Action Class",
            text_auto='.1f' # Show the exact percentage on the bars
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, width = "stretch")
        
        st.divider()
        
        # --- ORIGINAL TIMELINE & LOGS ---
        st.subheader("⏱️ Live Timeline")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Line chart showing confidence over time
            fig_line = px.line(df, x="time", y="confidence", color="action", title="Confidence Timeline", markers=True)
            st.plotly_chart(fig_line, width = "stretch")
            
        with col2:
            # The raw data table
            st.dataframe(df, width = "stretch", hide_index=True)
            
                # Clear button to reset the session state
            if st.button("🗑️ Clear History", width = "stretch"):
                st.session_state.history = []
                st.rerun()
    else:
        st.write("")
        st.info("No actions logged yet. Go to the Live Stream or Upload Video page to generate data.")
# ==========================================
# PAGE 5: SETTINGS
# ==========================================
elif page == "Settings":
    st.title("⚙️ Threshold Settings")
    st.session_state.confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, max_value=1.0, 
        value=st.session_state.confidence_threshold, 
        step=0.05
    )
    st.caption("Lowering this makes the system more sensitive but increases false positives.")