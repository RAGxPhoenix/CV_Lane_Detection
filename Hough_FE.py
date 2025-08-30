'''import streamlit as st
import sqlite3
import time
import os
import json
import requests
import uuid
import tempfile
import cv2
import numpy as np
from datetime import datetime
from urllib.parse import urlencode
from pathlib import Path

# -----------------------------
# Insert your existing processing code (process_frame and helpers)
# -----------------------------
# Copy your process_frame, roi_mask, average_and_extrapolate here.
# I'll include your functions verbatim (slightly adapted for import inside Streamlit).
def roi_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    pts = np.array([[
        (int(0.1*w), h),
        (int(0.4*w), int(0.6*h)),
        (int(0.6*w), int(0.6*h)),
        (int(0.9*w), h)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, pts, (255,)*img.shape[2])
    return cv2.bitwise_and(img, mask)

def average_and_extrapolate(lines, img_shape, min_slope=0.3):
    if lines is None: return None, None
    left, right = [], []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        if x2 == x1: continue
        m = (y2-y1)/(x2-x1)
        if abs(m) < min_slope: continue
        b = y1 - m*x1
        if m < 0:
            left.append((m,b))
        else:
            right.append((m,b))
    h = img_shape[0]
    y1 = h
    y2 = int(h*0.6)
    def make_line(arr):
        if not arr: return None
        m = np.mean([a for a,_ in arr])
        b = np.mean([b for _,b in arr])
        x1 = int((y1 - b) / m)
        x2 = int((y2 - b) / m)
        return (x1,y1,x2,y2)
    return make_line(left), make_line(right)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges_roi = roi_mask(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    edges_roi_gray = cv2.cvtColor(edges_roi, cv2.COLOR_BGR2GRAY)

    lines = cv2.HoughLinesP(edges_roi_gray, 1, np.pi/180, threshold=20,
                            minLineLength=40, maxLineGap=20)

    left_line, right_line = average_and_extrapolate(lines, frame.shape)

    out = frame.copy()
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            cv2.line(out, (x1,y1), (x2,y2), (0,0,255), 2)

    if left_line:
        cv2.line(out, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0,255,0), 8)
    if right_line:
        cv2.line(out, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0,255,0), 8)

    if left_line and right_line:
        ml = (left_line[3]-left_line[1])/(left_line[2]-left_line[0])
        bl = left_line[1] - ml*left_line[0]
        mr = (right_line[3]-right_line[1])/(right_line[2]-right_line[0])
        br = right_line[1] - mr*right_line[0]
        bottom_y = frame.shape[0]
        left_x = int((bottom_y - bl)/ml)
        right_x = int((bottom_y - br)/mr)
        lane_center = int((left_x + right_x)/2)
        veh_center = frame.shape[1]//2
        cv2.circle(out,(lane_center,bottom_y-10),6,(255,0,255),-1)
        cv2.circle(out,(veh_center,bottom_y-10),6,(0,165,255),-1)
        dev_pix = veh_center - lane_center
        thresh = int(0.05 * frame.shape[1])
        state = "WARN" if abs(dev_pix) > thresh else "OK"
        cv2.putText(out, f"Deviation: {dev_pix}px [{state}]", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255) if state=="WARN" else (0,255,0), 3)
    else:
        cv2.putText(out, "Lane detection unreliable", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    return out, edges_roi_gray

# -----------------------------
# Small helper utilities
# -----------------------------
DB_PATH = "uploads.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            name TEXT,
            filename TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn

def insert_upload(conn, email, name, filename):
    c = conn.cursor()
    c.execute("INSERT INTO uploads (email, name, filename, timestamp) VALUES (?, ?, ?, ?)",
              (email, name, filename, datetime.utcnow().isoformat()))
    conn.commit()

# -----------------------------
# Google OAuth helpers
# -----------------------------
CLIENT_SECRETS_FILE = "client_secret.json"  # download from Google Cloud and place here
REDIRECT_URI = "http://localhost:8501/"

def load_client_secrets():
    if not os.path.exists(CLIENT_SECRETS_FILE):
        return None
    with open(CLIENT_SECRETS_FILE, "r") as f:
        data = json.load(f)
    # standard downloaded JSON from Google stores client_id under installed or web
    creds = data.get("installed") or data.get("web")
    return creds

def build_auth_url(client_id, state):
    params = {
        "client_id": client_id,
        "response_type": "code",
        "scope": "openid email profile",
        "redirect_uri": REDIRECT_URI,
        "state": state,
        "access_type": "offline",
        "prompt": "consent"
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

def exchange_code_for_tokens(client_id, client_secret, code):
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }
    r = requests.post(token_url, data=data, timeout=10)
    r.raise_for_status()
    return r.json()

def get_userinfo_from_idtoken(id_token):
    # Use Google's tokeninfo endpoint (simple)
    r = requests.get("https://oauth2.googleapis.com/tokeninfo", params={"id_token": id_token}, timeout=10)
    r.raise_for_status()
    return r.json()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Lane Hough Demo", layout="wide")
st.title("Lane-Detection (Hough) — Upload & Live Preview")

# init db
conn = init_db()

# Load Google client secrets (must exist)
client_info = load_client_secrets()
if client_info is None:
    st.error("Missing client_secret.json. Create OAuth credentials in Google Console and put client_secret.json here.")
    st.stop()

client_id = client_info["client_id"]
client_secret = client_info["client_secret"]

# Authentication area
st.sidebar.header("Sign In")
if "auth_state" not in st.session_state:
    st.session_state.auth_state = {}

params = st.experimental_get_query_params()
if "code" in params and "state" in params:
    # user returned from Google's consent screen
    code = params["code"][0]
    state = params["state"][0]
    saved_state = st.session_state.get("oauth_state")
    if saved_state is None or state != saved_state:
        st.sidebar.error("OAuth state mismatch. Try signing in again.")
    else:
        try:
            tok = exchange_code_for_tokens(client_id, client_secret, code)
            id_token = tok.get("id_token")
            info = get_userinfo_from_idtoken(id_token)
            # save user in session
            st.session_state.user = {
                "email": info.get("email"),
                "name": info.get("name"),
                "picture": info.get("picture")
            }
            # clear query params to avoid re-processing code on reload
            st.experimental_set_query_params()
            st.sidebar.success(f"Signed in as {st.session_state.user['email']}")
        except Exception as e:
            st.sidebar.error(f"Auth failed: {e}")

if "user" not in st.session_state:
    # show sign-in button (link)
    if st.sidebar.button("Start Google Sign-In"):
        state = str(uuid.uuid4())
        st.session_state.oauth_state = state
        auth_url = build_auth_url(client_id, state)
        st.sidebar.markdown(f"[Click here to authenticate with Google]({auth_url})")
    st.sidebar.info("After completing consent, you'll be redirected back to this app.")
    st.stop()

# At this point user is signed in
user = st.session_state.user
st.sidebar.image(user.get("picture", ""), width=80) if user.get("picture") else None
st.sidebar.markdown(f"**{user.get('name','-')}**")
st.sidebar.markdown(f"`{user.get('email','-')}`")

# Upload area
st.header("Upload lane video (mp4, mov, avi, mkv)")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4","mov","avi","mkv"])

if uploaded_file is not None:
    # Save upload temporarily
    tdir = tempfile.gettempdir()
    tfpath = os.path.join(tdir, f"upload_{int(time.time())}_{uploaded_file.name}")
    with open(tfpath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved to {tfpath}")
    # insert into DB
    insert_upload(conn, user["email"], user.get("name",""), uploaded_file.name)
    st.info("Upload recorded in the local database.")

    # Controls
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        start = st.button("Start Processing")
    with col2:
        stop = st.button("Stop Processing")
    with col3:
        st.write("Processed output will be saved as: processed_<filename>.mp4")

    if "processing" not in st.session_state:
        st.session_state.processing = False

    if start:
        st.session_state.processing = True
    if stop:
        st.session_state.processing = False

    # Placeholder for live frames and progress
    frame_placeholder = st.empty()
    status = st.empty()
    progress_bar = st.progress(0)

    # Prepare output video writer (match input properties)
    cap_probe = cv2.VideoCapture(tfpath)
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    out_name = f"processed_{Path(uploaded_file.name).stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_name, fourcc, fps, (w,h))

    # Process loop
    if st.session_state.processing:
        cap = cv2.VideoCapture(tfpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        current = 0
        status.info("Processing... press 'Stop Processing' to interrupt")
        try:
            while cap.isOpened() and st.session_state.processing:
                ret, frame = cap.read()
                if not ret:
                    break
                out_frame, edges = process_frame(frame)
                out_writer.write(out_frame)
                # convert BGR->RGB for display
                frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                current += 1
                if total_frames > 0:
                    progress = min(1.0, current / total_frames)
                    progress_bar.progress(progress)
                # small sleep to allow Streamlit UI refresh and not hog CPU (adjust as needed)
                time.sleep(max(0.001, 1.0 / (fps or 20.0)))
            status.success("Processing finished.")
        except Exception as e:
            status.error(f"Processing stopped: {e}")
        finally:
            cap.release()
            out_writer.release()
            st.session_state.processing = False
            progress_bar.progress(1.0)
            st.success(f"Saved processed video as `{out_name}`")
            st.video(out_name)  # show finished video
else:
    st.info("Please sign in and upload a video to start.")'''


# streamlit_app.py
import streamlit as st
import sqlite3
import time
import os
import tempfile
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# -----------------------------
# Lane detection functions (your pipeline)
# -----------------------------
def roi_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    pts = np.array([[
        (int(0.1*w), h),
        (int(0.4*w), int(0.6*h)),
        (int(0.6*w), int(0.6*h)),
        (int(0.9*w), h)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, pts, (255,)*img.shape[2])
    return cv2.bitwise_and(img, mask)

def average_and_extrapolate(lines, img_shape, min_slope=0.3):
    if lines is None: return None, None
    left, right = [], []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        if x2 == x1: continue
        m = (y2-y1)/(x2-x1)
        if abs(m) < min_slope: continue
        b = y1 - m*x1
        if m < 0:
            left.append((m,b))
        else:
            right.append((m,b))
    h = img_shape[0]
    y1 = h
    y2 = int(h*0.6)
    def make_line(arr):
        if not arr: return None
        m = np.mean([a for a,_ in arr])
        b = np.mean([b for _,b in arr])
        # safety: avoid division by zero
        if abs(m) < 1e-6:
            return None
        x1 = int((y1 - b) / m)
        x2 = int((y2 - b) / m)
        return (x1,y1,x2,y2)
    return make_line(left), make_line(right)

def process_frame(frame):
    """
    Input: BGR frame (numpy array)
    Output: (annotated_frame (BGR), edges_roi_gray)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges_roi = roi_mask(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    edges_roi_gray = cv2.cvtColor(edges_roi, cv2.COLOR_BGR2GRAY)

    lines = cv2.HoughLinesP(edges_roi_gray, 1, np.pi/180, threshold=20,
                            minLineLength=40, maxLineGap=20)

    left_line, right_line = average_and_extrapolate(lines, frame.shape)

    out = frame.copy()
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            cv2.line(out, (x1,y1), (x2,y2), (0,0,255), 2)

    if left_line:
        cv2.line(out, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0,255,0), 8)
    if right_line:
        cv2.line(out, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0,255,0), 8)

    if left_line and right_line:
        # compute lane bottom intersections and deviation
        ml = (left_line[3]-left_line[1])/(left_line[2]-left_line[0])
        bl = left_line[1] - ml*left_line[0]
        mr = (right_line[3]-right_line[1])/(right_line[2]-right_line[0])
        br = right_line[1] - mr*right_line[0]
        bottom_y = frame.shape[0]
        # avoid division by zero
        if abs(ml) < 1e-6 or abs(mr) < 1e-6:
            cv2.putText(out, "Lane detection unreliable", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            return out, edges_roi_gray
        left_x = int((bottom_y - bl)/ml)
        right_x = int((bottom_y - br)/mr)
        lane_center = int((left_x + right_x)/2)
        veh_center = frame.shape[1]//2
        cv2.circle(out,(lane_center,bottom_y-10),6,(255,0,255),-1)
        cv2.circle(out,(veh_center,bottom_y-10),6,(0,165,255),-1)
        dev_pix = veh_center - lane_center
        thresh = int(0.05 * frame.shape[1])
        state = "WARN" if abs(dev_pix) > thresh else "OK"
        cv2.putText(out, f"Deviation: {dev_pix}px [{state}]", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255) if state=="WARN" else (0,255,0), 3)
    else:
        cv2.putText(out, "Lane detection unreliable", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    return out, edges_roi_gray

# -----------------------------
# Simple DB helpers (SQLite)
# -----------------------------
DB_PATH = "uploads.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            name TEXT,
            filename TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn

def insert_upload(conn, email, name, filename):
    c = conn.cursor()
    c.execute("INSERT INTO uploads (email, name, filename, timestamp) VALUES (?, ?, ?, ?)",
              (email, name, filename, datetime.utcnow().isoformat()))
    conn.commit()

def list_user_uploads(conn, email):
    c = conn.cursor()
    c.execute("SELECT id, filename, timestamp FROM uploads WHERE email=? ORDER BY id DESC", (email,))
    return c.fetchall()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Lane Hough Demo", layout="wide")
st.title("Lane-Detection (Hough) — Upload & Live Preview")

conn = init_db()

# --------- Mock / Dev Sign-in (no Google) ----------
# Quick dev login so you can test without OAuth or Google billing
if "user" not in st.session_state:
    st.sidebar.header("Dev Sign-In (for testing)")
    name = st.sidebar.text_input("Name", value="Dev User")
    email = st.sidebar.text_input("Email", value="dev@example.com")
    if st.sidebar.button("Sign in (dev)"):
        st.session_state.user = {"name": name, "email": email, "picture": ""}
        st.sidebar.success(f"Signed in as {email}")
if "user" not in st.session_state:
    st.info("Use the Dev Sign-In (sidebar) to test uploading and processing.")
    st.stop()
# ---------------------------------------------------

user = st.session_state.user
st.sidebar.markdown(f"**{user.get('name','-')}**")
st.sidebar.markdown(f"`{user.get('email','-')}`")

st.header("Upload lane video (mp4, mov, avi, mkv)")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4","mov","avi","mkv"])

# Processing controls
col1, col2, col3 = st.columns([1,1,2])
with col1:
    start_clicked = st.button("Start Processing")
with col2:
    stop_clicked = st.button("Stop Processing")
with col3:
    st.write("Processed output will be saved as: processed_<filename>.mp4")

if "processing" not in st.session_state:
    st.session_state.processing = False

# Stop button logic
if stop_clicked:
    st.session_state.processing = False

# If file uploaded, save to temp and record in DB
temp_filepath = None
if uploaded_file is not None:
    # Save uploaded bytes to a temp file on disk
    tdir = tempfile.gettempdir()
    temp_filepath = os.path.join(tdir, f"upload_{int(time.time())}_{uploaded_file.name}")
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved upload to: {temp_filepath}")
    insert_upload(conn, user["email"], user.get("name",""), uploaded_file.name)
    st.info("Upload recorded in local DB.")

# Show list of this user's uploads
st.sidebar.markdown("### Your uploads")
rows = list_user_uploads(conn, user["email"])
if rows:
    for r in rows[:10]:
        st.sidebar.markdown(f"- `{r[1]}` at {r[2]}")
else:
    st.sidebar.markdown("_No uploads yet_")

# Live preview & processing
placeholder = st.empty()
progress_bar = st.progress(0.0)
status_txt = st.empty()

if start_clicked and temp_filepath:
    st.session_state.processing = True

if temp_filepath and st.session_state.processing:
    # Prepare writer
    cap_probe = cv2.VideoCapture(temp_filepath)
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_probe.release()

    out_name = f"processed_{Path(uploaded_file.name).stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_name, fourcc, fps, (w,h))
    cap = cv2.VideoCapture(temp_filepath)

    status_txt.info("Processing... (press Stop Processing to interrupt)")

    try:
        frame_idx = 0
        while cap.isOpened() and st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                break
            out_frame, edges = process_frame(frame)
            out_writer.write(out_frame)

            # convert to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            frame_idx += 1
            if total_frames > 0:
                progress = min(1.0, frame_idx / total_frames)
                progress_bar.progress(progress)
            else:
                # If total unknown, show a pulsing bar
                progress_bar.progress((frame_idx % 100) / 100.0)

            # Give the UI time to update & not hog CPU
            # Adjust sleep depending on fps / desired speed
            time.sleep(max(0.001, 1.0 / (fps or 20.0)))

        status_txt.success("Processing finished.")
    except Exception as e:
        status_txt.error(f"Processing error: {e}")
    finally:
        cap.release()
        out_writer.release()
        st.session_state.processing = False
        progress_bar.progress(1.0)
        placeholder.empty()
        st.success(f"Saved processed video as: `{out_name}`")
        # Show resulting video
        st.video(out_name)

elif temp_filepath:
    st.info("Ready to process. Click **Start Processing** to begin.")
else:
    st.info("Upload a video file to enable processing controls.")
