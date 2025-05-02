import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import mediapipe as mp

# เตรียมโมเดล
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

with open("face_svm_model.pkl", "rb") as f:
    face_model = joblib.load("face_svm_model.pkl")

# ฟังก์ชันฝั่งนับนิ้วแบบใช้ landmark
def count_fingers(landmarks):
    tips_ids = [4, 8, 12, 16, 20]  # ปลายนิ้วหัวแม่มือถึงนิ้วก้อย
    fingers = []

    # หัวแม่มือ
    if landmarks.landmark[tips_ids[0]].x < landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # นิ้วชี้ถึงนิ้วก้อย
    for i in range(1, 5):
        if landmarks.landmark[tips_ids[i]].y < landmarks.landmark[tips_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

def get_face_embedding(pil_img):
    face = mtcnn(pil_img)
    if face is not None:
        emb = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        return emb
    return None

# Streamlit UI
st.title("🔐 ตรวจใบหน้า + ✋ นับนิ้วด้วย Landmark")

face_file = st.file_uploader("📤 อัปโหลดภาพใบหน้า (.jpg/.png)")
if face_file:
    img = Image.open(face_file).convert("RGB")
    st.image(img, caption="Uploaded Face", use_column_width=True)

    emb = get_face_embedding(img)
    if emb is not None:
        pred = face_model.predict([emb])[0]
        st.success(f"✅ เป็นสมาชิก: {pred}")

        # อัปโหลดรูปมือ
        hand_file = st.file_uploader("✋ อัปโหลดภาพมือเพื่อนับนิ้ว", type=["jpg", "jpeg", "png"])
        if hand_file:
            file_bytes = np.asarray(bytearray(hand_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils

            with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        count = count_fingers(hand_landmarks)
                        st.success(f"🖐️ ตรวจพบนิ้วจำนวน: {count} นิ้ว")

                        annotated_img = rgb.copy()
                        mp_drawing.draw_landmarks(
                            annotated_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )
                        st.image(annotated_img, caption=f"{count} Fingers", use_column_width=True)
                else:
                    st.warning("❌ ไม่พบมือในภาพ")
    else:
        st.error("❌ ไม่พบใบหน้าในภาพที่อัปโหลด")
else:
    st.info("กรุณาอัปโหลดภาพใบหน้า")
