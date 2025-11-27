import streamlit as st
import cv2, numpy as np, mediapipe as mp, tensorflow as tf
from collections import deque

st.set_page_config(page_title="Sign Language to Text", layout="wide")

# ---------------- UI Header ----------------
st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>🤟 Sign Language → Text Converter</h1>
<p style='text-align:center; font-size:18px;'>Choose Alphabets or Words & start performing gestures!</p>
""", unsafe_allow_html=True)

mode = st.selectbox("Select Recognition Mode", ["Word Recognition", "Alphabet Recognition"])

start_btn = st.button("▶ Start Camera")
stop_btn = st.button("⛔ Stop")

placeholder = st.empty()

# ---------------- Mediapipe ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- Load Models ----------------
if mode == "Word Recognition":
    model = tf.keras.models.load_model("model_lstm.h5")
    WORDS = ["book", "hello", "no", "thankyou", "yes"]
    SEQ_LENGTH, CONF_TH = 30, 0.65
    seq = deque(maxlen=SEQ_LENGTH)
    smooth = deque(maxlen=6)

else:
    model = tf.keras.models.load_model("asl_mnist_model.h5")
    CONF_TH = 0.75
    label_map = {i: chr(65+i) for i in range(25)}   # A-Z (no J)
    smooth = deque(maxlen=6)

# ---------------- Streamlit Camera Stream ----------------
if start_btn:
    cap = cv2.VideoCapture(0)
    st.info("Camera Started! Raise your hand to detect 🤚")

    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img)

            pred_text = "Waiting..."

            if mode == "Word Recognition":
                kpts = np.zeros(63)

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]

                if mode == "Word Recognition":
                    for i, lm in enumerate(hand.landmark):
                        kpts[i*3] = lm.x
                        kpts[i*3+1] = lm.y
                        kpts[i*3+2] = lm.z
                    seq.append(kpts)

                h,w,_ = frame.shape
                xs, ys = [], []
                for lm in hand.landmark:
                    xs.append(int(lm.x * w)); ys.append(int(lm.y * h))
                x1,y1 = max(min(xs)-20,0), max(min(ys)-20,0)
                x2,y2 = min(max(xs)+20,w), min(max(ys)+20,h)

                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                _, mask = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                # -------- Model Prediction ----------
                if mode == "Word Recognition" and len(seq)==SEQ_LENGTH:
                    preds = model.predict(np.array(seq).reshape(1,SEQ_LENGTH,63), verbose=0)[0]
                    idx, conf = np.argmax(preds), np.max(preds)
                    smooth.append(idx if conf>CONF_TH else "none")
                    most = max(set(smooth), key=smooth.count)
                    if most!="none": pred_text = WORDS[most]

                if mode == "Alphabet Recognition":
                    img = cv2.resize(mask,(28,28))/255.0
                    img = img.reshape(1,28,28,1)
                    preds = model.predict(img, verbose=0)[0]
                    idx, conf = np.argmax(preds), np.max(preds)
                    smooth.append(idx if conf>CONF_TH else "none")
                    most = max(set(smooth), key=smooth.count)
                    if most!="none": pred_text = label_map[most]

                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # ------------ Fancy Display Box ------------
            cv2.rectangle(frame, (10,10),(450,70),(0,0,0),-1)
            cv2.putText(frame, f"Prediction: {pred_text}", (20,55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

            placeholder.image(frame, channels="BGR")

    cap.release()
    st.warning("Camera stopped")

st.caption("Made with ❤️ using Streamlit, Mediapipe, TensorFlow")
