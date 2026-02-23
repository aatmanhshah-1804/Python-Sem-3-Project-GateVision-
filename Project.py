import streamlit as st
import sqlite3
import os
import cv2
from ultralytics import YOLO
from datetime import datetime

# =============================
# SQLite Database Setup
# =============================
DB_NAME = "people_counter.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """)

    # Records table
    c.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            video_name TEXT,
            in_count INTEGER,
            out_count INTEGER,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# =============================
# Authentication Functions
# =============================
def signup(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, password))
        conn.commit()
        return False, "Signup successful"
    except sqlite3.IntegrityError:
        return True, "User already exists"
    finally:
        conn.close()

def login(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, password))
    user = c.fetchone()
    conn.close()

    if user:
        return True, "Login successful"
    return False, "Invalid credentials"

# =============================
# YOLO PEOPLE COUNTER
# =============================
def people_counter(video_path):

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        st.error("Video not found or cannot be opened")
        return

    h, w, _ = frame.shape
    line_y = int(h * 0.60)
    margin = 5

    count_in = 0
    count_out = 0
    track_data = {}
    counted_ids = set()

    stframe = st.empty()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.4,
            iou=0.5
        )

        if results and results[0].boxes.id is not None:
            for box, tid in zip(results[0].boxes.xyxy, results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                tid = int(tid)

                if tid not in track_data:
                    track_data[tid] = {
                        "initial_side": "up" if cy < line_y else "down",
                        "prev_side": "up" if cy < line_y else "down"
                    }
                    continue

                prev_side = track_data[tid]["prev_side"]
                current_side = "up" if cy < line_y - margin else "down"

                if tid not in counted_ids:
                    if (
                        track_data[tid]["initial_side"] == "up"
                        and prev_side == "up"
                        and current_side == "down"
                    ):
                        count_in += 1
                        counted_ids.add(tid)

                    elif (
                        track_data[tid]["initial_side"] == "down"
                        and prev_side == "down"
                        and current_side == "up"
                    ):
                        count_out += 1
                        counted_ids.add(tid)

                track_data[tid]["prev_side"] = current_side

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)

        cv2.line(frame, (0, line_y), (w, line_y), (0,255,0), 2)
        cv2.putText(frame, f"IN: {count_in}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"OUT: {count_out}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()

    # SAVE RESULT TO DATABASE
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO records (username, video_name, in_count, out_count, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (
        st.session_state.username,
        os.path.basename(video_path),
        count_in,
        count_out,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

    st.success(f"Final Count → IN: {count_in} | OUT: {count_out}")

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="People Counter", layout="centered")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "signup"

st.title("🚶 People Counter System")

# ---------- SIGN UP ----------
if not st.session_state.authenticated and st.session_state.page == "signup":
    st.subheader("Sign Up")

    su_user = st.text_input("Username")
    su_pass = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        error, msg = signup(su_user, su_pass)
        if not error:
            st.session_state.authenticated = True
            st.session_state.username = su_user
            st.success(msg)
        else:
            st.error(msg)

    if st.button("Login"):
        st.session_state.page = "login"

# ---------- LOGIN ----------
elif not st.session_state.authenticated and st.session_state.page == "login":
    st.subheader("Login")

    lg_user = st.text_input("Username")
    lg_pass = st.text_input("Password", type="password")

    if st.button("Login"):
        success, msg = login(lg_user, lg_pass)
        if success:
            st.session_state.authenticated = True
            st.session_state.username = lg_user
            st.success(msg)
        else:
            st.error(msg)

    if st.button("Sign Up"):
        st.session_state.page = "signup"

# ---------- DASHBOARD ----------
# ---------- DASHBOARD ----------
else:
    st.sidebar.success(f"👤 Logged in as: {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.page = "signup"
        st.rerun()

    st.subheader("YOLO Based People Counter")

    video_path = st.text_input("Video file path", "people_test.mp4")

    if st.button("Start Counting"):
        people_counter(video_path)

    # 🔽 ADD THIS BLOCK HERE
    if st.button("📊 Show Database Records"):

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        st.subheader("👥 Registered Users")
        c.execute("SELECT username, password FROM users")
        users = c.fetchall()

        if users:
            st.table(users)
        else:
            st.info("No users found")

        st.subheader("📁 Counting Records")
        c.execute("""
            SELECT username, video_name, in_count, out_count, timestamp
            FROM records
        """)
        records = c.fetchall()

        if records:
            st.table(records)
        else:
            st.info("No records found")

        conn.close()
