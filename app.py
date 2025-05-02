import streamlit as st
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import cv2
import mediapipe as mp
from pathlib import Path
import math

# ------------------ SETUP ------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ------------------ LOAD MEMBERS ------------------
@st.cache_resource
def load_member_embeddings(member_dir):
    embeddings = []
    for img_path in Path("face_member").glob("*.jpg"):
        name = img_path.stem.split("_")[0]
        img = Image.open(img_path).convert("RGB")
        face = mtcnn(img)
        if face is not None:
            emb = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
            embeddings.append((name, emb))
    return embeddings

members = load_member_embeddings("face_member")

# ------------------ FINGER COUNT ------------------
def vector_angle(a, b, c):
    """คำนวณมุมที่จุด b ระหว่าง a-b-c"""
    ba = [a.x - b.x, a.y - b.y]
    bc = [c.x - b.x, c.y - b.y]

    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba * mag_bc == 0:
        return 0
    cosine = dot_product / (mag_ba * mag_bc)
    angle = math.acos(min(1, max(-1, cosine)))  # กัน NaN
    return math.degrees(angle)
    
def count_fingers(landmarks):
    # Indices ของข้อนิ้วกลางแต่ละนิ้ว
    fingers = [8, 12, 16, 20]  # นิ้วชี้ถึงก้อย
    mcp_ids = [5, 9, 13, 17]   # โคนแต่ละนิ้ว

    count = 0
    for tip, mcp in zip(fingers, mcp_ids):
        angle = vector_angle(landmarks.landmark[mcp], landmarks.landmark[mcp + 1], landmarks.landmark[tip])
        if angle < 160:  # ถ้ามุมแหลม แสดงว่านิ้วเหยียดตรง
            count += 1

    # ตรวจนิ้วโป้งแยก
    angle_thumb = vector_angle(landmarks.landmark[2], landmarks.landmark[3], landmarks.landmark[4])
    if angle_thumb < 160:
        count += 1

    return count

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
st.title("GESSURE: ยืนยันใบหน้า + ตรวจนิ้ว")

face_file = st.file_uploader("📤 อัปโหลดภาพใบหน้า (jpg/png)", type=["jpg", "jpeg", "png"])
if face_file:
    face_img = Image.open(face_file).convert("RGB")
    st.image(face_img, caption="Uploaded Face", use_container_width=True)

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
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=10, circle_radius=12),
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=10)
                        )
                        st.image(annotated_img, caption=f"{count} Fingers", use_container_width=True)
                else:
                    st.warning("❌ ไม่พบมือในภาพ")
    else:
        st.error("❌ ไม่พบในระบบสมาชิก")
else:
    st.info("📸 กรุณาอัปโหลดภาพใบหน้า")
