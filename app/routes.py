from flask import Blueprint, request, jsonify
from app.model import moderate_message

# Define a Blueprint for modular routes
moderation_bp = Blueprint("moderation", __name__)

@moderation_bp.route("/moderate", methods=["POST"])
def moderate():
    data = request.json
    message = data.get("message", "")
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    result = moderate_message(message)
    return jsonify(result)
