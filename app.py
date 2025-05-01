
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import mediapipe as mp
import os
from pathlib import Path
import streamlit as st

# 2. เตรียมโมเดล
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 3. ฟังก์ชัน
def get_face_embedding(pil_image):
    face = mtcnn(pil_image)
    if face is not None:
        face = face.unsqueeze(0).to(device)
        emb = resnet(face).detach().cpu().numpy()
        return emb[0]
    else:
        return None

def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

def load_members_embeddings(member_dir):
    member_embeddings = []
    for img_path in Path(member_dir).glob('*.jpg'):
        img = Image.open(img_path)
        name = img_path.stem.split('_')[0]  # ตัดชื่อเอาเฉพาะชื่อก่อน _ เช่น akshay_1.jpg → akshay
        emb = get_face_embedding(img)
        if emb is not None:
            member_embeddings.append((name, emb))
    return member_embeddings

# 4. แอปหลัก
def main():
    st.title("GesSure: ตรวจใบหน้าและนับนิ้วมือ")

    member_dir = 'face_member'
    member_embeddings = load_members_embeddings(member_dir)

    uploaded_file = st.file_uploader("📤 อัปโหลดภาพใบหน้าเพื่อตรวจสอบสมาชิก")
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='ภาพที่อัปโหลด', use_column_width=True)
        emb = get_face_embedding(img)

        if emb is not None:
            found = False
            for name, ref_emb in member_embeddings:
                sim = cosine_similarity([ref_emb], [emb])[0][0]
                if sim > 0.85:
                    st.success(f"🔐 เป็นสมาชิก: {name} (Similarity: {sim:.4f})")
                    found = True
                    break
            if not found:
                st.error("❌ ไม่ใช่สมาชิก")
        else:
            st.warning("⚠️ ไม่พบใบหน้าในภาพที่อัปโหลด")

    st.markdown("---")
    st.header("✋ ตรวจนับนิ้วจากภาพมือ")
    hand_img = st.file_uploader("📤 อัปโหลดภาพมือ", key='hand')
    if hand_img is not None:
        file_bytes = np.asarray(bytearray(hand_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                finger_count = count_fingers(lm)
                st.success(f"🖐️ ตรวจพบนิ้วจำนวน: {finger_count} นิ้ว")
        else:
            st.warning("ไม่พบมือในภาพ")

if __name__ == '__main__':
    main()
