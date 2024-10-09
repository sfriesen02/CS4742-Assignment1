import numpy as np
import pandas as pd

if __name__ == '__main__':

    # imports raw data sets
    train = pd.read_csv("https://raw.githubusercontent.com/MuhammedBuyukkinaci/TensorFlow-Sentiment-Analysis-on-Amazon-Reviews-Data/refs/heads/master/dataset/train_amazon.csv")
    test = pd.read_csv("https://raw.githubusercontent.com/MuhammedBuyukkinaci/TensorFlow-Sentiment-Analysis-on-Amazon-Reviews-Data/refs/heads/master/dataset/test_amazon.csv")

    from sklearn.feature_extraction.text import CountVectorizer

    # CountVectorizer() converts text to a matrix of token counts
    vectorizer = CountVectorizer()

    # Fit CountVectorizer() onto training data and transform into a matrix of token counts
    x_train = vectorizer.fit_transform(train['text'])

    # Transform test data into matrix of tokens
    x_test = vectorizer.transform(test['text'])

    # Extracts the labels (0 for negative and 1 for positive) from the data sets
    y_train = train['label']
    y_test = test['label']

    # Imports necessary classes for logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import confusion_matrix, classification_report

    # Instantiate the classifier
    clf = OneVsRestClassifier(LogisticRegression())

    # Fit the classifier to the training data
    # Trains the model to understand the relationship between input features and labels
    clf.fit(x_train, y_train)

    # Print the accuracy of classifier on test data set
    # Compares predicted labels with actual labels
    print("Accuracy: {}".format(clf.score(x_test, y_test)))

    # Uses trained classifiers to predict the labels of test set
    x_test_clv_pred = clf.predict(x_test)

    # Prints detailed classification report
    print(classification_report(y_test, x_test_clv_pred, target_names=['negative', 'positive']))

    # Create and print confusion matrix
    cm = confusion_matrix(y_test, x_test_clv_pred)
    print(f"Confusion Matrix:\n{cm}")

