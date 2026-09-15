"""
generate_face_dataset.py — Generates a local synthetic face dataset.
Allows running the full VAE & GAN pipeline instantly without waiting for huge downloads.
Creates diverse synthetic faces with skin tones, hair, eyes, and expressions.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "faces", "train", "images")
NUM_IMAGES = 2000
IMG_SIZE = 64

def generate_faces():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    skin_tones = [
        (255, 224, 189),
        (234, 192, 134),
        (214, 156, 89),
        (165, 114, 62),
        (115, 74, 38),
        (80, 50, 25),
    ]

    hair_colors = [
        (20, 20, 20),
        (80, 50, 20),
        (160, 100, 40),
        (220, 180, 80),
        (140, 30, 20),
        (90, 90, 90),
    ]

    for i in range(NUM_IMAGES):
        # Base background
        bg_color = (random.randint(180, 230), random.randint(180, 230), random.randint(200, 240))
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=bg_color)
        draw = ImageDraw.Draw(img)

        skin = random.choice(skin_tones)
        hair = random.choice(hair_colors)

        # Head / face oval
        w_offset = random.randint(-2, 2)
        h_offset = random.randint(-2, 2)
        face_box = [16 + w_offset, 14 + h_offset, 48 + w_offset, 54 + h_offset]

        # Hair background
        hair_top = max(4, face_box[1] - random.randint(4, 8))
        draw.ellipse([face_box[0] - 3, hair_top, face_box[2] + 3, face_box[3] - 10], fill=hair)

        # Draw Face
        draw.ellipse(face_box, fill=skin)

        # Hair front / bangs
        draw.chord([face_box[0] - 1, hair_top, face_box[2] + 1, face_box[1] + 12], 0, 180, fill=hair)

        # Eyes
        eye_y = face_box[1] + 16
        eye_spacing = random.randint(8, 11)
        eye_w = random.randint(2, 3)
        eye_h = random.randint(2, 3)
        center_x = (face_box[0] + face_box[2]) // 2

        draw.ellipse([center_x - eye_spacing - eye_w, eye_y - eye_h,
                      center_x - eye_spacing + eye_w, eye_y + eye_h], fill=(30, 30, 30))
        draw.ellipse([center_x + eye_spacing - eye_w, eye_y - eye_h,
                      center_x + eye_spacing + eye_w, eye_y + eye_h], fill=(30, 30, 30))

        # Eyebrows
        brow_y = eye_y - 4
        draw.line([center_x - eye_spacing - eye_w - 1, brow_y, center_x - eye_spacing + eye_w + 1, brow_y], fill=hair, width=1)
        draw.line([center_x + eye_spacing - eye_w - 1, brow_y, center_x + eye_spacing + eye_w + 1, brow_y], fill=hair, width=1)

        # Nose
        nose_y = eye_y + 8
        draw.line([center_x, eye_y + 4, center_x, nose_y], fill=(max(0, skin[0]-30), max(0, skin[1]-30), max(0, skin[2]-30)), width=1)

        # Mouth / smile
        mouth_y = nose_y + random.randint(5, 7)
        mouth_w = random.randint(4, 7)
        mouth_color = (max(0, skin[0]-50), max(0, skin[1]-60), max(0, skin[2]-50))
        draw.arc([center_x - mouth_w, mouth_y - 2, center_x + mouth_w, mouth_y + 3], 0, 180, fill=mouth_color, width=2)

        img.save(os.path.join(OUTPUT_DIR, f"face_{i:04d}.png"))

    print(f"✓ Generated {NUM_IMAGES} synthetic face images in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_faces()
