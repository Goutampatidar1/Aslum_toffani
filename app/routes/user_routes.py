from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.services.user_service import create_user, get_user, list_all_user, delete_user
import logging
import os
import uuid 
import shutil

# from dotenv import load_dotenv
# load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)

user_bp = Blueprint("user_bp", __name__)

UPLOAD_FOLDER = "app/images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_files(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# route for adading user
@user_bp.route("/user", methods=["POST"])
def add_user():
    if "image" not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    image = request.files["image"]
    data = request.form.to_dict()
    logging.info(data)

    if not all(k in data for k in ("name", "email_id", "contact_number", "major")):
        return jsonify({"error": "Missing required fields"}), 400

    if image.filename == "" or not allowed_files(image.filename):
        return jsonify({"error": "Invalid or empty image file"}), 400

    generated_id = uuid.uuid1()
    original_ext = os.path.splitext(image.filename)[1]  # Get file extension
    new_filename = f"{generated_id}{original_ext}"
    image_path = os.path.join(UPLOAD_FOLDER, new_filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    try:
        image.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500

    try:
        user_id, error = create_user(data, image_path, generated_id)
        if error:
            if os.path.exists(image_path):
                os.remove(image_path)
            return jsonify({"error": "User Already Exist "}), 403
        return (
            jsonify({"message": "User created and image processed", "id": user_id}),
            201,
        )
    except ValueError as e:
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({"error": str(e)}), 409
    except Exception as e:
        if os.path.exists(image_path):
            os.remove(image_path)
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


# @user_bp.route("/user", methods=["POST"])
# def add_user():
#     if "image" not in request.files:
#         return jsonify({"error": "Image files are required"}), 400

#     images = request.files.getlist("image")  # Expecting multiple images
#     data = request.form.to_dict()
#     logging.info(data)

#     if not all(k in data for k in ("name", "email_id", "contact_number", "major" , "company_id")):
#         return jsonify({"error": "Missing required fields"}), 400

#     for image in images:
#         if image.filename == "" or not allowed_files(image.filename):
#             return jsonify({"error": f"Invalid image file: {image.filename}"}), 400

#     # Generate UUID and create a directory for this user
#     generated_id = str(uuid.uuid1())
#     user_folder = os.path.join(UPLOAD_FOLDER, generated_id)
#     os.makedirs(user_folder, exist_ok=True)

#     image_paths = []
#     try:
#         for image in images:
#             ext = os.path.splitext(image.filename)[1]
#             filename = f"{uuid.uuid4()}{ext}"
#             image_path = os.path.join(user_folder, filename)
#             image.save(image_path)
#             image_paths.append(image_path)
#     except Exception as e:
#         return jsonify({"error": f"Failed to save images: {str(e)}"}), 500

#     try:
#         user_id, error = create_user(data, image_paths, generated_id)
#         if error:
#             shutil.rmtree(user_folder, ignore_errors=True)  # Clean up folder
#             return jsonify({"error": "User Already Exist"}), 403
#         return jsonify({
#             "message": "User created and images processed",
#             "id": user_id
#         }), 201

#     except ValueError as e:
#         shutil.rmtree(user_folder, ignore_errors=True)
#         return jsonify({"error": str(e)}), 409

#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#         shutil.rmtree(user_folder, ignore_errors=True)
#         return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500




# route for getting details for particular user
@user_bp.route("/user/<user_id>", methods=["GET"])
def fetch_user_by_id(user_id):
    user, error = get_user(user_id)
    if error:
        return jsonify({"error": error}), 404
    return jsonify(user), 200


# route for getting all the user
@user_bp.route("/get-all-user", methods=["GET"])
def fetch_all_user():
    users, error = list_all_user()
    if error:
        return jsonify({"error": error}), 400
    return jsonify(users), 200


@user_bp.route("/user/<user_id>", methods=["DELETE"])
def remove_user(user_id):
    delete_count, error = delete_user(user_id)
    if error:
        return jsonify({"error": error}), 404
    return jsonify({"message": "User deleted successfully"}), 200
