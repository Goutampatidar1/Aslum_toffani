from flask import Blueprint, jsonify
from app.services.stream_services import start_stream, stop_stream

stream_bp = Blueprint("stream", __name__)

@stream_bp.route("/start-stream/<camera_id>", methods=["GET"])
def start_stream_route(camera_id):
    try:
        port, msg = start_stream(camera_id)
        return jsonify({"wsPort": port, "message": msg})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@stream_bp.route("/stop-stream/<camera_id>", methods=["GET"])
def stop_stream_route(camera_id):
    try:
        msg = stop_stream(camera_id)
        return jsonify({"message": msg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
