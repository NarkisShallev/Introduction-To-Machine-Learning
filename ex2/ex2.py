import sys
import numpy as np
from numpy import linalg as li


# min-max norm between 0-1
def min_max_norm(matrix_value):
    # computes minimum in each column
    old_min = np.min(matrix_value, axis=0)
    # computes maximum in each column
    old_max = np.max(matrix_value, axis=0)
    new_min = 0
    new_max = 1
    new_size = new_max - new_min
    for i in range(len(matrix_value)):
        for j in range(len(matrix_value[0])):
            if old_max[j] - old_min[j] == 0:
                break
            matrix_value[i][j] = ((matrix_value[i][j] - old_min[j]) / (old_max[j] - old_min[j])) * new_size + new_min

    return matrix_value


# z-score norm
def z_score_norm(matrix_value):
    matrix_value = (matrix_value - matrix_value.mean()) / matrix_value.std()
    return matrix_value


# perceptron algorithm
def perceptron_algorithm(train_x, train_y, eta, epochs):
    # create a matrix w consisting of 8 columns (for 8 features) and 3 rows (for 3 classes).
    num_of_features = len(train_x[0])
    w = np.asarray(np.zeros((3, num_of_features)))

    for e in range(epochs):
        for x, y in zip(train_x, train_y):
            # classifying the example by the w that returns the maximum classification
            y_hat = np.argmax(np.dot(w, x))
            # if the classifier was wrong
            if y != y_hat:
                # increase all the w's in the values at the y position and reduce
                # in the values at the y_hat position
                w[y, :] = w[y, :] + eta * x
                w[y_hat, :] = w[y_hat, :] - eta * x
    return w


# svm algorithm
def svm_algorithm(train_x, train_y, eta, lamda, epochs):
    # create a matrix w consisting of 8 columns (for 8 features) and 3 rows (for 3 classes).
    num_of_features = len(train_x[0])
    w = np.asarray(np.zeros((3, num_of_features)))

    for e in range(epochs):
        for x, y in zip(train_x, train_y):
            # classifying the example by the w that returns the maximum classification
            y_hat = np.argmax(np.dot(w, x))
            # if the classifier was wrong
            if y != y_hat:
                # increase all the w's in the values at the y position and reduce
                # in the values at the y_hat position
                w[y, :] = (1 - eta * lamda) * w[y, :] + eta * x
                w[y_hat, :] = (1 - eta * lamda) * w[y_hat, :] - eta * x
                w[3 - y - y_hat, :] = (1 - eta * lamda) * w[3 - y - y_hat, :]
    return w


# passive aggressive algorithm
def passive_aggressive_algorithm(train_x, train_y, epochs):
    # create a matrix w consisting of 8 columns (for 8 features) and 3 rows (for 3 classes).
    num_of_features = len(train_x[0])
    w = np.asarray(np.zeros((3, num_of_features)))

    for e in range(epochs):
        for x, y in zip(train_x, train_y):
            # classifying the example by the w that returns the maximum classification
            y_hat = np.argmax(np.dot(w, x))
            # l = loss(w,x,y) = max(0,1 - w_y*x^t + w_y_hat*x^t)
            l = max(0, 1 - (np.dot(w[y, :], x)) + (np.dot(w[y_hat, :], x)))
            # tau = l / (2*||x||^2)
            tau = l / (2 * (li.norm(x) ** 2))
            if li.norm(x) == 0:
                tau = l / (2 * (0.000001 ** 2))
            # if the classifier was wrong
            if y != y_hat:
                # increase all the w's in the values at the y position and reduce
                # in the values at the y_hat position
                w[y, :] = w[y, :] + tau * x
                w[y_hat, :] = w[y_hat, :] - tau * x
    return w


# classifying the examples by the w that returns the maximum class
def classify(train_x, w):
    classifications = []
    for x in train_x:
        classifications.append(np.argmax(np.dot(w, x)))
    return classifications


# count error percentages
def count_error_percentages(classifications, train_y):
    num_of_examples = len(train_y)
    if num_of_examples == 0:
        return 0
    count = 0
    for y_hat, y in zip(classifications, train_y):
        if y != y_hat:
            count = count + 1
    return (count / num_of_examples) * 100


# prepare train_x and test_x
def prepare_train_x_and_test_x(file):
    # reads all the rows from the file
    file_lines = open(file, 'r').readlines()
    data = []
    # removes enters, turns letters into numeric values, separates by commas and convert to float
    for line in range(len(file_lines)):
        line_without_enter = file_lines[line].replace('\n', '').replace("M", '0').replace("I", '1').replace("F", '2')
        data.append(np.asarray(line_without_enter.split(",")).astype(np.float))
    return np.asarray(data)


# prepare train_y
def prepare_train_y(file):
    # reads all the rows from the file
    file_lines = open(file, 'r').readlines()
    data = []
    # removes enters, turns letters into numeric values, separates by commas and convert to int
    for line in range(len(file_lines)):
        line_without_enter = file_lines[line].replace('\n', '')
        data.append(np.asarray(line_without_enter).astype(np.int))
    return np.asarray(data)


if len(sys.argv) < 4:
        exit(-1)

# prepare data
train_x = prepare_train_x_and_test_x(sys.argv[1])
train_y = prepare_train_y(sys.argv[2])
test_x = prepare_train_x_and_test_x(sys.argv[3])

# shuffle
np.random.seed(2)
even_spaced_values = np.arange(len(train_x))
np.random.shuffle(even_spaced_values)
train_x = train_x[even_spaced_values]
train_y = train_y[even_spaced_values]

# normalization
#train_x = min_max_norm(train_x)
#train_x = z_score_norm(train_x)

# prepare data for training
num_of_examples = len(train_x)
ninety_percent_num_of_examples = int(num_of_examples * 9 / 10)
ninety_percent_train_x = train_x[:ninety_percent_num_of_examples]
ninety_percent_train_y = train_y[:ninety_percent_num_of_examples]
ten_percent_train_x = train_x[ninety_percent_num_of_examples:]
ten_percent_train_y = train_y[ninety_percent_num_of_examples:]

epochs = 40
eta = 0.0001
lamda = 0.01

# perceptron training
w_perceptron = perceptron_algorithm(ninety_percent_train_x, ninety_percent_train_y, eta, epochs)
#classifications_perceptron = classify(ten_percent_train_x, w_perceptron)
#print(count_error_percentages(classifications_perceptron, ten_percent_train_y))

# svm training
w_svm = svm_algorithm(ninety_percent_train_x, ninety_percent_train_y, eta, lamda, epochs)
#classifications_svm = classify(ten_percent_train_x, w_svm)
#print(count_error_percentages(classifications_svm, ten_percent_train_y))

# passive aggressive training
w_passive_aggressive = passive_aggressive_algorithm(ninety_percent_train_x, ninety_percent_train_y, epochs)
#classifications_passive_aggressive = classify(ten_percent_train_x, w_passive_aggressive)
#print(count_error_percentages(classifications_passive_aggressive, ten_percent_train_y))

# classify test_x
classifications_perceptron = classify(test_x, w_perceptron)
classifications_svm = classify(test_x, w_svm)
classifications_passive_aggressive = classify(test_x, w_passive_aggressive)

for per, s, pa in zip(classifications_perceptron, classifications_svm, classifications_passive_aggressive):
    print(f"perceptron: {per}, svm: {s}, pa: {pa}")
