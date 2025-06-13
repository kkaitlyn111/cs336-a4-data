import joblib
import numpy as np

# Load the saved vectorizer and classifier
vectorizer = joblib.load("/home/user/cs336-a4-data/cs336_data/qualityclassifier/vectorizer.joblib")
classifier = joblib.load("/home/user/cs336-a4-data/cs336_data/qualityclassifier/classifier.joblib")

def classify_quality(text):
    text = text.replace('\n', ' ')
    X = vectorizer.transform([text])
    label = classifier.predict(X)[0]
    if label == '0':
        label = 'cc'
    else:
        label = 'wiki'
    prob = np.max(classifier.predict_proba(X))
    return label, prob


