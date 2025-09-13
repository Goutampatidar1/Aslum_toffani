from pymongo import MongoClient, errors
import os
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", None)
DATABASE_NAME = os.getenv("DATABASE_NAME", "my-database")

print(f"DATABASE NAME : {DATABASE_NAME} , DATABASE_URL : {DATABASE_URL}")

try:
    client = MongoClient(DATABASE_URL, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = [DATABASE_NAME]
    print("DATABSE CONNNECTED SUCCESSFULLY")

    try:
        db.users.create_index("email_id", unique=True)
        db.cameras.create_index("url", unique=True)
        print("Unique index on camera_name and camera_place created successfully.")
    except errors.OperationFailure as e:
        print(f"Failed to create index: {e}")

except errors.ServerSelectionTimeoutError as ex:
    print("Failed to Connect to database")
    raise SystemExit(1)
