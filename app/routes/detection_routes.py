from flask import Blueprint, request, jsonify
import threading
import logging
from pathlib import Path
from app.detection_app.app import CameraStream

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


# Store running streams
active_streams = {}
streams_lock = threading.Lock()

detection_bp = Blueprint("detection_bp", __name__)


def validate_camera_payload(data):
    required = [
        "camera_id",
        "rtsp_url",
        # "emb_db_path",
        # "checkin_cooldown",
        # "checkout_cooldown",
        # "show_window",
    ]
    missing = [field for field in required if data.get(field) is None]
    if missing:
        return False, f"Missing required field(s): {', '.join(missing)}"
    return True, None


@detection_bp.route("/start_camera", methods=["POST"])
def start_camera():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON payload"}), 400

    valid, msg = validate_camera_payload(data)
    if not valid:
        return jsonify({"error": msg}), 400

    camera_id = data["camera_id"]
    with streams_lock:
        if camera_id in active_streams:
            return jsonify({"message": f"Camera {camera_id} is already running"}), 200

        try:
            emb_db_path = data["emb_db_path"]
            if not Path(emb_db_path).exists():
                return (
                    jsonify({"error": f"Embedding DB file not found at {emb_db_path}"}),
                    400,
                )

            stream = CameraStream(
                camera_id=camera_id,
                rtsp_url=data["rtsp_url"],
                # emb_db_path=emb_db_path,
                # checkin_cooldown=int(data["checkin_cooldown"]),
                # checkout_cooldown=int(data["checkout_cooldown"]),
                # show_window=bool(data["show_window"]),
                # use_gpu=bool(data.get("use_gpu", True)),
            )

            thread = threading.Thread(target=stream.run, daemon=True)
            active_streams[camera_id] = {"stream": stream, "thread": thread}
            thread.start()

            logging.info(f"[API] Started camera {camera_id}")
            return jsonify({"message": f"Camera {camera_id} started"}), 200

        except Exception as e:
            logging.error(f"Error starting camera {camera_id}: {e}")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500


@detection_bp.route("/stop_camera", methods=["POST"])
def stop_camera():
    data = request.get_json()
    if not data or "camera_id" not in data:
        return jsonify({"error": "Missing camera_id"}), 400

    camera_id = data["camera_id"]
    with streams_lock:
        if camera_id not in active_streams:
            return jsonify({"error": f"Camera {camera_id} is not running"}), 404

        stream = active_streams[camera_id]["stream"]
        thread = active_streams[camera_id]["thread"]

        try:
            stream.stop()
            thread.join(timeout=5)
            del active_streams[camera_id]
            logging.info(f"[API] Stopped camera {camera_id}")
            return jsonify({"message": f"Camera {camera_id} stopped"}), 200
        except Exception as e:
            logging.error(f"Error stopping camera {camera_id}: {e}")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500


@detection_bp.route("/status", methods=["GET"])
def status():
    with streams_lock:
        cameras = list(active_streams.keys())
    return jsonify({"running_cameras": cameras}), 200
