from app.config import db
from bson.objectid import ObjectId
from utils.helper import str_object_id
from pymongo.errors import DuplicateKeyError
from app.models.user_model import User
import os


# function for listing all the user
def list_all_user():
    users = list(db.users.find({}))
    for user in users:
        user["_id"] = str(users["_id"])
    return users


# function for creating the user
def create_user(user_data, image_path):
        user = User(
            name=user_data["name"],
            email_id=user_data["email_id"],
            contact_number=user_data["contact_number"],
            image=image_path,
        )

        existing_user = db.users.find_one({"email_id": user.email_id})
        if existing_user:
            raise ValueError("Email id already Exist")

        result = db.users.insert_one(user.to_dict())
        user_id = str(result.inserted_id)

        # # Have to run this check Later
        # encoded = encode_face(image_path, user.name)
        # if not encoded:
        #     # Optionally, you can delete the user here if encoding fails
        #     db.users.delete_one({"_id": result.inserted_id})
        #     raise ValueError("No face detected in the uploaded image")

        return user_id , None


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

    # regenerate_embeddings()

    return result.deleted_count, None
