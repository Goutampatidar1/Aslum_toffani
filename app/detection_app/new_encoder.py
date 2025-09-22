# import os
# import cv2
# import pickle
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path
# from insightface.app import FaceAnalysis
# import logging

# EMB_DB_PATH = Path("encodings/embeddings.pkl")
# IMAGES_ROOT = Path("images")
# DET_SIZE = (640, 640)
# USE_GPU = True
# MAX_IMAGES_PER_PERSON = 3

# logging.basicConfig(
#     level=logging.INFO,
#     format="[%(asctime)s] %(levelname)s - %(message)s"
# )


# def init_face_app(use_gpu=True, det_size=(640, 640)):
#     providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
#     app = FaceAnalysis(name="buffalo_l", providers=providers)
#     app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)
#     return app


# def load_embeddings(db_path: Path):
#     if db_path.exists():
#         try:
#             with open(db_path, "rb") as f:
#                 return pickle.load(f)
#         except Exception as e:
#             logging.error(f"Failed to load embeddings: {e}")
#     return []


# def save_embeddings(embeddings, db_path: Path):
#     db_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(db_path, "wb") as f:
#         pickle.dump(embeddings, f)
#     logging.info(f"Embeddings saved to: {db_path}")

# def encode_all_faces(images_root: Path, db_path: Path, max_images: int = 3):
#     if not images_root.exists():
#         logging.error(f"Image folder not found: {images_root}")
#         return False

#     face_app = init_face_app(USE_GPU, DET_SIZE)
#     existing_embeddings = load_embeddings(db_path)
#     updated_embeddings = []

#     for person_dir in tqdm(images_root.iterdir(), desc="Processing users"):
#         if not person_dir.is_dir():
#             continue

#         user_id = person_dir.name
#         logging.info(f"Processing user: {user_id}")
#         person_embeddings = []

#         image_files = sorted([
#             f for f in person_dir.iterdir()
#             if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
#         ])[:max_images]

#         if not image_files:
#             logging.warning(f"No images found for {user_id}")
#             continue

#         for img_path in image_files:
#             img = cv2.imread(str(img_path))
#             if img is None:
#                 logging.warning(f"Failed to read image: {img_path}")
#                 continue

#             faces = face_app.get(img)
#             if not faces:
#                 logging.warning(f"No face detected in: {img_path}")
#                 continue

#             largest_face = max(
#                 faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
#             )
#             emb = largest_face.normed_embedding.astype(np.float32)
#             person_embeddings.append(emb)

#         if not person_embeddings:
#             logging.warning(f"No valid face embeddings for user: {user_id}")
#             continue

#         avg_embedding = np.mean(person_embeddings, axis=0)

#         # Remove any existing entry for this user
#         existing_embeddings = [e for e in existing_embeddings if e["name"] != user_id]

#         updated_embeddings.append({
#             "name": user_id,
#             "embedding": avg_embedding
#         })

#         logging.info(f"Encoded {len(person_embeddings)} image(s) for {user_id}")


#     final_embeddings = existing_embeddings + updated_embeddings
#     save_embeddings(final_embeddings, db_path)

#     logging.info("Encoding complete for all users.")
#     return True


# if __name__ == "__main__":
#     success = encode_all_faces(IMAGES_ROOT, EMB_DB_PATH, MAX_IMAGES_PER_PERSON)
#     if success:
#         logging.info("Finished face encoding pipeline.")
#     else:
#         logging.error("Face encoding pipeline failed.")



import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from insightface.app import FaceAnalysis
import logging

EMB_DB_PATH = Path("encodings/embeddings.pkl")
IMAGES_ROOT = Path("images")
DET_SIZE = (640, 640)
USE_GPU = True
MAX_IMAGES_PER_PERSON = 3

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)

def init_face_app(use_gpu=True, det_size=(640, 640)):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    try:
        app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)
        logging.info("InsightFace model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load InsightFace model: {e}")
        raise
    return app

def load_embeddings(db_path: Path):
    if db_path.exists():
        try:
            with open(db_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Failed to load embeddings: {e}")
    return []

def save_embeddings(embeddings, db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, "wb") as f:
        pickle.dump(embeddings, f)
    logging.info(f"Embeddings saved to: {db_path}")

def encode_all_faces(images_root: Path, db_path: Path, max_images: int = 3):
    if not images_root.exists():
        logging.error(f"Image folder not found: {images_root}")
        return False

    face_app = init_face_app(USE_GPU, DET_SIZE)
    existing_embeddings = load_embeddings(db_path)
    updated_embeddings = []

    for person_dir in tqdm(images_root.iterdir(), desc="Processing users"):
        if not person_dir.is_dir():
            continue

        user_id = person_dir.name
        logging.info(f"Processing user: {user_id}")
        person_embeddings = []

        image_files = sorted([
            f for f in person_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ])[:max_images]

        if not image_files:
            logging.warning(f"No images found for {user_id}")
            continue

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning(f"Failed to read image: {img_path}")
                continue
            else:
                logging.info(f"Loaded image: {img_path}")

            faces = face_app.get(img)
            logging.info(f"Faces detected in {img_path}: {len(faces)}")
            if not faces:
                logging.warning(f"No face detected in: {img_path}")
                continue

            # Find largest face by bbox area
            largest_face = max(
                faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            emb = largest_face.normed_embedding.astype(np.float32)
            person_embeddings.append(emb)

        if not person_embeddings:
            logging.warning(f"No valid face embeddings for user: {user_id}")
            continue

        avg_embedding = np.mean(person_embeddings, axis=0)

        # Remove existing entry for this user before updating
        existing_embeddings = [e for e in existing_embeddings if e["name"] != user_id]

        updated_embeddings.append({
            "name": user_id,
            "embedding": avg_embedding
        })

        logging.info(f"Encoded {len(person_embeddings)} image(s) for {user_id}")

    final_embeddings = existing_embeddings + updated_embeddings
    save_embeddings(final_embeddings, db_path)

    logging.info("Encoding complete for all users.")
    return True

if __name__ == "__main__":
    success = encode_all_faces(IMAGES_ROOT, EMB_DB_PATH, MAX_IMAGES_PER_PERSON)
    if success:
        logging.info("Finished face encoding pipeline.")
    else:
        logging.error("Face encoding pipeline failed.")
