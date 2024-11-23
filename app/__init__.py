from flask import Flask
from app.routes import moderation_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(moderation_bp)
    return app
