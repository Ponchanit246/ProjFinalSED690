import streamlit as st
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import cv2
import mediapipe as mp
from pathlib import Path

# ------------------ SETUP ------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ------------------ LOAD MEMBERS ------------------
@st.cache_resource
def load_member_embeddings(member_dir):
    embeddings = []
    for img_path in Path(member_dir).glob("*.jpg"):
        name = img_path.stem.split("_")[0]
        img = Image.open(img_path).convert("RGB")
        face = mtcnn(img)
        if face is not None:
            emb = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
            embeddings.append((name, emb))
    return embeddings

members = load_member_embeddings("face_member")

# ------------------ FINGER COUNT ------------------
def count_fingers(lm):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    if lm.landmark[tips_ids[0]].x < lm.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for i in range(1, 5):
        if lm.landmark[tips_ids[i]].y < lm.landmark[tips_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

# ------------------ FACE VERIFY ------------------
def verify_face(pil_image, members, threshold=0.80):
    face = mtcnn(pil_image)
    if face is None:
        return None, 0.0
    emb = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
    for name, ref_emb in members:
        sim = cosine_similarity([emb], [ref_emb])[0][0]
        if sim > threshold:
            return name, sim
    return None, 0.0

# ------------------ STREAMLIT UI ------------------
st.title("🔐 GESSURE: ยืนยันใบหน้า + ตรวจนิ้ว")

face_file = st.file_uploader("📤 อัปโหลดภาพใบหน้า (jpg/png)", type=["jpg", "jpeg", "png"])
if face_file:
    face_img = Image.open(face_file).convert("RGB")
    st.image(face_img, caption="Uploaded Face", use_column_width=True)

    name, sim = verify_face(face_img, members)
    if name:
        st.success(f"✅ เป็นสมาชิก: {name} (Similarity: {sim:.2f})")

        # 👋 Upload hand
        hand_file = st.file_uploader("✋ อัปโหลดภาพมือ", type=["jpg", "jpeg", "png"])
        if hand_file:
            file_bytes = np.asarray(bytearray(hand_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    for lm in result.multi_hand_landmarks:
                        count = count_fingers(lm)
                        st.success(f"🖐️ ตรวจพบนิ้วจำนวน: {count} นิ้ว")

                        annotated_img = rgb.copy()
                        mp_drawing.draw_landmarks(
                            annotated_img, lm, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )
                        st.image(annotated_img, caption=f"{count} Fingers", use_column_width=True)
                else:
                    st.warning("❌ ไม่พบมือในภาพ")
    else:
        st.error("❌ ไม่พบในระบบสมาชิก")
else:
    st.info("📸 กรุณาอัปโหลดภาพใบหน้า")
