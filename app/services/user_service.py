from app.config import db
from bson.objectid import ObjectId
from app.utils.helper import str_object_id
from app.detection_app.face_encoder import encode_face
from app.models.user_model import User
import os
from datetime import datetime
import shutil
from app.detection_app.new_encoder import encode_all_faces
from pathlib import Path


def serialize_user(user):
    def serialize(value):
        if isinstance(value, ObjectId):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, list):
            return [serialize(v) for v in value]
        elif isinstance(value, dict):
            return {k: serialize(v) for k, v in value.items()}
        else:
            return value

    return {key: serialize(val) for key, val in user.items()}


# function for listing all the user
def list_all_user():
    users = list(db.users.find({}))
    serialized_users = [serialize_user(user) for user in users]
    return serialized_users, None


# # function for creating the user
# def create_user(user_data, image_path , generated_id):
#     user = User(
#         name=user_data["name"],
#         email_id=user_data["email_id"],
#         contact_number=user_data["contact_number"],
#         image=image_path,
#         major=user_data["major"],
#         unique_user_id= str(generated_id)
#     )

#     existing_user = db.users.find_one({"email_id": user.email_id})
#     if existing_user:
#         return 0, "User Already exist"

#     result = db.users.insert_one(user.to_dict())
#     user_id = str(result.inserted_id)

#     # Encoder implemented
#     success = encode_face(image_path, user.unique_user_id)
#     if not success:
#         db.users.delete_one({"_id": result.inserted_id})
#         if os.path.exists(image_path):
#             os.remove(image_path)
#         raise ValueError("No face detected in the uploaded image")

#     return user_id, None


def create_user(user_data, image_paths, generated_id):
    user = User(
        name=user_data["name"],
        email_id=user_data["email_id"],
        contact_number=user_data["contact_number"],
        images=image_paths,  # <- now passing full list
        major=user_data["major"],
        unique_user_id=generated_id,
        company_id=user_data["company_id"]
    )

    existing_user = db.users.find_one({"email_id": user.email_id})
    if existing_user:
        return 0, "User Already exist"

    result = db.users.insert_one(user.to_dict())
    user_id = str(result.inserted_id)

    # success = encode_all_faces(Path("images"), Path("encodings/embeddings.pkl"))
    user_folder = Path(image_paths[0]).parent
    success = encode_all_faces(user_folder, Path("encodings/embeddings.pkl"))

    if not success:
        db.users.delete_one({"_id": result.inserted_id})
        shutil.rmtree(os.path.dirname(image_paths[0]), ignore_errors=True)
        raise ValueError("No valid faces detected in uploaded images")

    return user_id, None


# function for getting particular user details
def get_user(user_id):
    object_id = str_object_id(user_id)
    if not object_id:
        return None
    user = db.users.find_one({"_id": object_id})

    if user:
        user["_id"] = str(user["_id"])
        return user, None
    return None, "User not found"


# function for updating the user
def update_user(user_id):
    pass


# function for deleting the user and regenerate the embeddings
def delete_user(user_id):
    object_id = str_object_id(user_id)
    if not object_id:
        return 0, "Invalid User ID format"

    user = db.users.find_one({"_id": object_id})
    if not user:
        return 0, "User Not found"

    image_path = user.get("image")
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except Exception as e:
            print(f"Error deleting image file {image_path}: {e}")

    result = db.users.delete_one({"_id": object_id})
    if result.deleted_count == 0:
        return 0, "User not found"

    # Regenrte the embeddings

    return result.deleted_count, None
