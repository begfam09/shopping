import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    print('Data loaded without error')
    x_train, x_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(x_train, y_train)
    print('Model created and trained.')
    predictions = model.predict(x_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    dict_month_int = {
        'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
        'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        evidence = []
        labels = []
        for row in reader:
            evidence.append([int(row[0]),    # Administrative Page
                             float(row[1]),  # Administrative Duration
                             int(row[2]),    # Informational Page
                             float(row[3]),  # Informational Duration
                             int(row[4]),    # Product Related Page
                             float(row[5]),  # Product Related Duration
                             float(row[6]),  # BounceRates
                             float(row[7]),  # ExitRates
                             float(row[8]),  # PageValues
                             float(row[9]),  # Special Day
                             dict_month_int[row[10]],  # Month
                             int(row[11]),   # Operating Systems
                             int(row[12]),   # Browser
                             int(row[13]),   # Region
                             int(row[14]),   # Traffic Type
                             int(row[15] == 'Returning_Visitor'),  # VisitorType
                             int(row[16] == 'TRUE')])  # Weekend
            labels.append(int(row[17] == 'TRUE'))  # Revenue

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

    """
    true_positives = 0
    predicted_positives = 0

    true_negatives = 0
    predicted_negatives = 0

    for i in range(0, len(labels)):
        if labels[i] == 1:
            true_positives += 1
            if predictions[i] == 1:
                predicted_positives += 1
        else:
            true_negatives += 1
            if predictions[i] == 0:
                predicted_negatives += 1

    return predicted_positives / true_positives, predicted_negatives / true_negatives
    """


if __name__ == "__main__":
    main()
