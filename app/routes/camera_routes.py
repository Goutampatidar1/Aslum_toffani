from flask import Blueprint, request, jsonify
from app.services.camera_service import create_camera, delete_camera, fetch_all_camera


camera_bp = Blueprint("camera_bp", __name__)


@camera_bp.route("/add-camera", methods=["POST"])
def add_camera():
    data = request.form.to_dict()

    if not all(k in data for k in ("url", "camera_name", "camera_place")):
        return jsonify({"error": "Missing fields"}), 400

    try:
        camera_id, error = create_camera(data)
        if error:
            return jsonify({"error": "Error while adding the camera"}), 400
        return (
            jsonify({"message": "Camera Added", "id": camera_id}),
            201,
        )
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@camera_bp.route("/delete/<camera_id>", methods=["DELETE"])
def remove_camera(camera_id):
    delete_count, error = delete_camera(camera_id)
    if error:
        return jsonify({"error": "error while delteing the camera"}), 400
    return jsonify({"message": "Camera Deleted succesfully"}), 200


@camera_bp.route("/all-camera", methods=["GET"])
def get_all_cameras():
    cameras, error = fetch_all_camera()
    if error:
        return jsonify({"error": error}), 400
    return jsonify(cameras), 200
