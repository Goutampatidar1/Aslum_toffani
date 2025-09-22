import pickle
from pathlib import Path

db_path = Path("../../encodings/embeddings.pkl")
if not db_path.exists():
    print("Embeddings file not found!")
else:
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    if not db:
        print("Embeddings database is empty!")
    else:
        print(f"Loaded {len(db)} entries:")
        for entry in db:
            print(entry["name"], entry["embedding"].shape)
