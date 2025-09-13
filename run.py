from flask import Flask
from app.routes.user_routes import user_bp

app = Flask(__name__)
app.register_blueprint(user_bp)

@app.errorhandler(404)
def not_found(error):
    return {"error": "Not found"}, 404

@app.errorhandler(500)
def internal_error(error):
    return {"error": "Internal server error"}, 500

if __name__ == "__main__":
    app.run(debug=True)
