from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# pre-trained model for chat moderation
MODEL_NAME = "Hate-speech-CNERG/dehatebert-mono-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def moderate_message(message):
    """
    Evaluate a message for moderation.
    Args:
        message (str): The chat message to evaluate.
    Returns:
        dict: Detailed moderation result with probabilities and raw logits.
    """
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    class_index = torch.argmax(probabilities, dim=1).item()

    class_labels = model.config.id2label
    human_label = class_labels[class_index] if class_labels else f"class_{class_index}"

    probabilities_dict = {
        model.config.id2label[idx] if class_labels else f"class_{idx}": prob.item()
        for idx, prob in enumerate(probabilities[0])
    }

    return {
        "predicted_label": human_label,
        "predicted_index": class_index,
        "confidence": probabilities[0][class_index].item(),
        "probabilities": probabilities_dict,
        "logits": logits.tolist(),
    }
