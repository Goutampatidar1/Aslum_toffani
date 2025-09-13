import logging
from flask import Flask
from threading import Thread
import asyncio
from app.routes.user_routes import user_bp
from app.routes.camera_routes import camera_bp
from app.services.stream_services import run_websocket_server
from app.routes.stream_routes import stream_bp
from app.routes.attendance_routes import attendance_bp

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)

app = Flask(__name__)
app.register_blueprint(user_bp)
app.register_blueprint(camera_bp)
app.register_blueprint(stream_bp)
app.register_blueprint(attendance_bp)


@app.errorhandler(404)
def not_found(error):
    return {"error": "Not found"}, 404


@app.errorhandler(500)
def internal_error(error):
    return {"error": "Internal server error"}, 500


def start_ws_server():
    """
    Starts the WebSocket server in a separate thread.
    Catches exceptions and logs errors without crashing.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        logging.info("Starting WebSocket server...")
        loop.run_until_complete(run_websocket_server())
        loop.run_forever()
    except Exception as e:
        logging.error(f"WebSocket server encountered an error: {e}")
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        logging.info("WebSocket server shut down cleanly.")


if __name__ == "__main__":
    try:
        ws_thread = Thread(target=start_ws_server, daemon=True)
        ws_thread.start()
        logging.info("WebSocket thread started.")

        logging.info("Starting Flask app...")
        app.run(host="127.0.0.1" , debug=True, use_reloader=False)

    except Exception as e:
        logging.error(f"Flask encountered an error: {e}")
    finally:
        logging.info("Application shutting down.")
