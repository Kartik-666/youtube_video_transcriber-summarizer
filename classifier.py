# import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# from torch.nn.functional import softmax

# # Initialize model and tokenizer globally
# MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"
# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
# model.eval()

# # Label mapping
# LABEL_MAP = {
#     0: "Request",
#     1: "Statement", 
#     2: "Question",
#     3: "Objection",
#     4: "Command",
#     5: "Explanation"
# }

# def classify_sentence(text):
#     """Standalone classification function that maintains your original interface"""
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = softmax(outputs.logits, dim=1)
#         pred_label_id = torch.argmax(probs, dim=1).item()
#     return LABEL_MAP.get(pred_label_id, "Unknown")






# import torch
# import re
# from transformers import BertTokenizer, BertForSequenceClassification
# from torch.nn.functional import softmax

# from sklearn.metrics import accuracy_score, classification_report


# tokenizer = BertTokenizer.from_pretrained('bhadresh-savani/bert-base-uncased-emotion')
# model = BertForSequenceClassification.from_pretrained('bhadresh-savani/bert-base-uncased-emotion')
# model.eval()

# # Enhanced dialogue act mapping
# DIALOGUE_ACT_MAP = {
#     0: "Offer/Request",  # "Can I help you?", "Would you like..."
#     1: "Statement",      # Declarative sentences
#     2: "Question",       # Questions
#     3: "Warning",        # Cautions
#     4: "Intent",         # "I'm gonna...", future plans
#     5: "Clarification"   # Explanations
# }

# def classify_sentence(text):
#     text = text.strip()
#     clean_text = re.sub(r'[^\w\s]', '', text.lower())
    
#     # Strong pattern matches first
#     if re.match(r'^(can i|may i|shall i)', clean_text):
#         return "Offer/Request"
    
#     if text.endswith('?'):
#         return "Question"
    
#     if any(text.lower().startswith(w) for w in ['don\'t', 'warning', 'be careful']):
#         return "Warning"
    
#     if re.match(r'^(i\'m gonna|i\'ll|i will|i\'m going to)', clean_text):
#         return "Intent"
    
#     # Model classification
#     inputs = tokenizer(text, return_tensors='pt', truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = softmax(outputs.logits, dim=1)
#         pred_label_id = torch.argmax(probs).item()
    
#     # Post-processing
#     label = DIALOGUE_ACT_MAP.get(pred_label_id, "Statement")
    
#     # Contextual overrides
#     if label == "Warning" and not any(w in clean_text for w in ['toxic', 'danger', 'careful']):
#         return "Statement"
    
#     return label
#  def evaluate_classifier():
#     from test_samples import test_samples
    
#     true_labels = []
#     pred_labels = []
    
#     for text, true_label in test_samples:
#         pred_label = classify_sentence(text)
#         true_labels.append(true_label)
#         pred_labels.append(pred_label)
    
#     accuracy = accuracy_score(true_labels, pred_labels)
#     report = classification_report(true_labels, pred_labels)
    
#     print(f"\nClassification Accuracy: {accuracy:.2%}")
#     print("\nDetailed Report:")
#     print(report)
    
#     return accuracy
# if __name__ == "__main__":
#     evaluate_classifier()
#====================================================================================================
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, classification_report

# Initialize model and tokenizer globally (fixes the tokenizer error)
tokenizer = BertTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
model = BertForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
model.eval()

# Original label mapping
LABEL_MAP = {
    0: "Question",
    1: "Statement",
    2: "Request",
    3: "Instruction",
    4: "Agreement",
    5: "Disagreement",
    6: "Emotion",
    7: "Clarification"
}

# def classify_sentence(text):
#     """Original classification function with tokenizer fix"""
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = softmax(outputs.logits, dim=1)
#         pred_label_id = torch.argmax(probs, dim=1).item()
#     return LABEL_MAP.get(pred_label_id, "Unknown")
def classify_sentence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
    return LABEL_MAP[pred], float(probs[0][pred])

# # Example usage
# print(f"[{label}] ({confidence:.0%})")  # [Request] (92%)


# Evaluation function (maintaining your test structure)
def evaluate_classifier(test_samples):
    true_labels = []
    pred_labels = []
    
    for text, true_label in test_samples:
        pred_label, _ = classify_sentence(text)
        true_labels.append(true_label)
        pred_labels.append(pred_label)
    
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)
    
    print(f"Classification Accuracy: {accuracy:.2%}")
    print("\nDetailed Report:")
    print(report)
    
    return accuracy

# from transformers import Trainer, TrainingArguments

# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=16,
#     num_train_epochs=2,
#     save_steps=100,
#     logging_dir="./logs"
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"]
# )
# trainer.train()

# Only run evaluation if executed directly
if __name__ == "__main__":
    from test_samples import test_samples  # Your test samples
    evaluate_classifier(test_samples)
#---------------=========================================----------------------
