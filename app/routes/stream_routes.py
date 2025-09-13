from flask import Blueprint, request, jsonify
import asyncio
import logging
from app.services.stream_services import start_stream, stop_stream

stream_bp = Blueprint('stream', __name__)

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s')


@stream_bp.route('/start_stream', methods=['POST'])
def start_stream_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON payload"}), 400

    camera_id = data.get("camera_id")
    camera_url = data.get("camera_url")
    ws_port = data.get("ws_port")

    if not camera_id or not camera_url or not ws_port:
        return jsonify({"error": "camera_id, camera_url, and ws_port are required"}), 400

    try:
        ws_port = int(ws_port)
    except ValueError:
        return jsonify({"error": "ws_port must be an integer"}), 400

    try:
        # Call the async start_stream function safely
        port = asyncio.run(start_stream(camera_id, camera_url, ws_port))
        if port is None:
            return jsonify({"error": "Failed to start stream"}), 500
        return jsonify({"message": "Stream started", "ws_port": port}), 200
    except Exception as e:
        logging.error(f"Error starting stream: {e}")
        return jsonify({"error": "Internal server error"}), 500


@stream_bp.route('/stop_stream', methods=['POST'])
def stop_stream_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON payload"}), 400

    camera_id = data.get("camera_id")
    if not camera_id:
        return jsonify({"error": "camera_id is required"}), 400

    try:
        # Call the async stop_stream function safely
        asyncio.run(stop_stream(camera_id))
        return jsonify({"message": "Stream stopped"}), 200
    except Exception as e:
        logging.error(f"Error stopping stream: {e}")
        return jsonify({"error": "Internal server error"}), 500
