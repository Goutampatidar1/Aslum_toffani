import cv2
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from insightface.app import FaceAnalysis
import logging


logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


EMB_DB = Path("encodings/embeddings.pkl")
EMB_DB.parent.mkdir(parents=True, exist_ok=True)
DET_SIZE = (640, 640)
USE_GPU = True


def init_app(use_gpu=True, det_size=(640, 640)):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)
    return app


def encode_face(image_path, label):
    """
    Encodes a single image and appends/updates it into the embeddings database.
    Returns True if successful, False otherwise.
    """
    try:
        app = init_app(USE_GPU, DET_SIZE)
        img = cv2.imread(str(image_path))
        if img is None:
            logging.error(f"Could not read image: {image_path}")
            return False

        faces = app.get(img)
        if not faces:
            logging.warning(f"No face detected in {image_path}")
            return False

        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        emb = face.normed_embedding.astype(np.float32)

        # Load existing embeddings if available
        embeddings = []
        if EMB_DB.exists():
            with open(EMB_DB, "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []

        # Remove old embeddings for this label if they exist
        embeddings = [entry for entry in embeddings if entry["name"] != label]

        # Append new embedding
        embeddings.append({"name": label, "embedding": emb})

        # Aggregate embeddings (optional, here it's just stored as is)
        with open(EMB_DB, "wb") as f:
            pickle.dump(embeddings, f)

        logging.info(f"Face encoding successful for {label}")
        return True  
    except Exception as e:
        logging.error(f"Error encoding face for {label}: {e}")
        return False
