from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.services.user_service import create_user, get_user, list_all_user, delete_user
import os

# from dotenv import load_dotenv
# load_dotenv()

user_bp = Blueprint("user_bp", __name__)

UPLOAD_FOLDER = "app/images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_files(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# route for adading user
@user_bp.route("/users", methods=["POST"])
def add_user():
    if "image" not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    image = request.files["image"]
    data = request.form.to_dict()

    if not all(k in data for k in ("name", "emailId", "phoneNumber")):
        return jsonify({"error": "Missing required fields"}), 400

    if image.filename == "" or not allowed_files(image.filename):
        return jsonify({"error": "Invalid or empty image file"}), 400

    filename = secure_filename(image.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        image.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500

    try:
        user_id, error = create_user(data, image_path)
        # have to check this
        if error:
            raise Exception
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
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


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
