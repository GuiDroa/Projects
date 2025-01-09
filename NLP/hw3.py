import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import numpy as np
from scipy.spatial.distance import cosine as cosDist

# Load the Hugging Face model and tokenizer
model_path = "/Users/gui/Desktop/Python/saved_model2"
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Initialize a list to store subreddits based on classification indices
subreddits = []

# Dummy model to align with grader function
def setModel(dummy_model):
    print("Model is set (dummy function for compatibility with HW3_grader)")

# Get predicted class ID for a comment
def comment_to_class(comment):
    inputs = tokenizer(comment, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return predicted_class_id

# Modified findPlagiarism using cosine similarity (based on pre-trained Hugging Face model embeddings)
def sentence_to_vec(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits[0].numpy()

def findPlagiarism(sentences, target):
    target_vec = sentence_to_vec(target)  # Vectorize target sentence
    similarities = []

    for sentence in sentences:
        sentence_vec = sentence_to_vec(sentence)  # Vectorize each sentence
        # Compute similarity or set to -1 if vector is zero
        if np.linalg.norm(target_vec) > 0 and np.linalg.norm(sentence_vec) > 0:
            similarity = 1 - cosDist(target_vec, sentence_vec)
        else:
            similarity = -1
        similarities.append(similarity)

    return np.argmax(similarities)  # Return index of highest similarity

# Training function - builds the subreddit mapping
def classifySubreddit_train(trainFile):
    global subreddits
    with open(trainFile, 'r') as file:
        data = [json.loads(line) for line in file]

    subreddit_map = {}
    index = 0

    # Map each subreddit to a unique index
    for item in data:
        subreddit = item['subreddit']
        if subreddit not in subreddit_map:
            subreddit_map[subreddit] = index
            index += 1

    # Create a list of subreddits based on the map
    subreddits = [None] * len(subreddit_map)
    for sub, idx in subreddit_map.items():
        subreddits[idx] = sub

    print("Training completed. Subreddit mapping created.")

# Test function - uses the transformer model to predict subreddit
def classifySubreddit_test(comment):
    predicted_class_id = comment_to_class(comment)
    return subreddits[predicted_class_id]  # Return the predicted subreddit
