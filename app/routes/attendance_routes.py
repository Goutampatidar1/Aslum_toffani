from flask import Blueprint, request, jsonify
from app.services.attendance_service import mark_attendance
import logging

attendance_bp = Blueprint("attendance", __name__)

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


@attendance_bp.route("/mark_attendance", methods=["POST"])
def mark_attendance_route():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        action = data.get("action")

        if not user_id or not action:
            return jsonify({"message": "User ID and action are required"}), 400

        attendance, error = mark_attendance(user_id, action)
        if error:
            return jsonify({"message": error}), (
                400 if error != "Internal server error" else 500
            )

        return (
            jsonify(
                {
                    "message": f"{action.capitalize()} successful",
                    "attendance": attendance,
                }
            ),
            200,
        )

    except Exception as e:
        logging.error(f"Exception in mark_attendance_route: {e}")
        return jsonify({"error": "Internal server error"}), 500
