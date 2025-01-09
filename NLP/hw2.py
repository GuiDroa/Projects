import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Global variables
vectorizer = CountVectorizer(ngram_range=(3, 6))  # Use trigrams to 6-grams (to reduce random chance)
model = MultinomialNB(alpha=1.0)  # Laplace smoothing


#Problem 1
def calcNGrams_train(trainFile):
    with open(trainFile, 'r', encoding='utf-8') as f: # Open File read
        data = f.readlines()

    # Process sentences
    sentences = [line.strip() for line in data if line.strip()]

    # Create n-gram features
    X = vectorizer.fit_transform(sentences).toarray()

    # Label 1 for regular sentences
    y = [1] * len(sentences)

    # Train the model
    model.fit(X, y)


def calcNGrams_test(sentences):
    # Use the trained model to get predictions
    X_test = vectorizer.transform(sentences).toarray()

    # Count N-gram matches
    match_counts = [0] * len(sentences)  # Initialize match counts

    for i in range(len(sentences)):
        match_counts[i] = sum(X_test[i])  # Count matches for each sentence

    # Find the index of the sentence with the lower number of matches
    min_match_index = match_counts.index(min(match_counts))
    
    return min_match_index  # Return index of the random sentence


# Problem 2
def calcSentiment_train(trainFile):
    global vectorizer, model # Global variables
    
    reviews = [] # Arrays
    sentiments = []

    # Read the training data from the jsonlist file
    with open(trainFile, 'r') as f:
        for line in f:
            data = json.loads(line)
            reviews.append(data['review']) # Extract reviews
            sentiments.append(data['sentiment']) #Extract sentiment

    # Transform the reviews into n-grams using vectorizer
    X_train = vectorizer.fit_transform(reviews)
    y_train = sentiments

    # Train the model
    model.fit(X_train, y_train)

def calcSentiment_test(review):
    global vectorizer, model # Global variables

    # Transform the review into the same feature space as the training data
    X_test = vectorizer.transform([review])
    
    # Predict the sentiment (either true or false)
    prediction = model.predict(X_test) 

    return bool(prediction[0]) 


