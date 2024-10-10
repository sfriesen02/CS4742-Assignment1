import pandas as pd
import time

if __name__ == '__main__':

    # imports raw data sets
    train = pd.read_csv("https://raw.githubusercontent.com/MuhammedBuyukkinaci/TensorFlow-Sentiment-Analysis-on-Amazon-Reviews-Data/refs/heads/master/dataset/train_amazon.csv")
    test = pd.read_csv("https://raw.githubusercontent.com/MuhammedBuyukkinaci/TensorFlow-Sentiment-Analysis-on-Amazon-Reviews-Data/refs/heads/master/dataset/test_amazon.csv")

    # Change training set size to measure impact on accuracy
    sample_train = train.sample(n=80000)

    from sklearn.feature_extraction.text import CountVectorizer

    # CountVectorizer() converts text to a matrix of token counts
    vectorizer = CountVectorizer()

    # Fit CountVectorizer() onto training data and transform into a matrix of token counts
    x_train = vectorizer.fit_transform(sample_train['text'])

    # Transform test data into matrix of tokens
    x_test = vectorizer.transform(test['text'])

    # Extracts the labels (0 for negative and 1 for positive) from the data sets
    y_train = sample_train['label']
    y_test = test['label']

    # Imports necessary classes for logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, classification_report

    # Different activation functions are used for comparison
    sigmoid_model = LogisticRegression()
    softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    # Trains the model to understand the relationship between input features and labels
    # Time() is used to measure how long the model trains for
    start_time = time.time()
    sigmoid_model.fit(x_train, y_train)
    sig_train_time = time.time() - start_time

    start_time = time.time()
    softmax_model.fit(x_train, y_train)
    soft_train_time = time.time() - start_time

    print(f"Sigmoid training time: {sig_train_time} seconds")
    print(f"Softmax training time: {soft_train_time} seconds")
    print()

    # Uses trained classifiers to predict the labels of test set
    # Time() is used to measure how long the model makes predictions
    start_time = time.time()
    x_test_sig_pred = sigmoid_model.predict(x_test)
    sig_inf_time = time.time() - start_time

    start_time = time.time()
    x_test_soft_pred = softmax_model.predict(x_test)
    soft_inf_time = time.time() - start_time

    print(f"Sigmoid Inference Time: {sig_inf_time} seconds")
    print(f"Softmax Inference Time: {soft_inf_time} seconds")
    print()

    # Compares predicted labels with actual labels
    print("Accuracy (Sigmoid): {}".format(sigmoid_model.score(x_test, y_test)))
    print("Accuracy (Softmax): {}".format(softmax_model.score(x_test,y_test)))
    print()

    # Prints detailed classification report
    print("Sigmoid Classification Report")
    print(classification_report(y_test, x_test_sig_pred, target_names=['negative', 'positive']))
    print("Softmax Classification Report")
    print(classification_report(y_test, x_test_soft_pred, target_names=['negative', 'positive']))

    # Create and print confusion matrix
    cm1 = confusion_matrix(y_test, x_test_sig_pred)
    cm2 = confusion_matrix(y_test, x_test_soft_pred)
    print(f"Confusion Matrix (Sigmoid):\n{cm1}")
    print(f"Confusion Matrix (Softmax):\n{cm2}")
    print()

    sig_efficiency = sig_train_time + sig_inf_time
    print("Efficiency (Sigmoid):", sig_efficiency, " seconds")
    soft_efficiency = soft_train_time + soft_inf_time
    print("Efficiency (Softmax):", soft_efficiency, " seconds")

