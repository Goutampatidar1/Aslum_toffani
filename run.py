import logging
from flask import Flask
from flask_cors import CORS

# Routes
from app.routes.user_routes import user_bp
from app.routes.camera_routes import camera_bp
from app.routes.stream_routes import stream_bp
from app.routes.attendance_routes import attendance_bp

def create_app():
    app = Flask(__name__)
    CORS(app)

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

    return app

def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    app = create_app()
    logging.info("[Main] Starting Flask app...")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

if __name__ == "__main__":
    main()
