import sys
import numpy as np

HIDDEN_LAYER_SIZE = 120
NUMBER_OF_PIXELS = 28*28
NUMBER_OF_CLASSES = 10


def z_score_norm(matrix_value):
    matrix_value = (matrix_value - matrix_value.mean()) / matrix_value.std()
    return matrix_value


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / exp.sum()


def ReLU_for_vector(x):
    return np.maximum(0, x)


def derivative_ReLU_for_vector(x):
    for i in range(len(x)):
        if x[i] > 0:
            x[i] = 1
        else:
            x[i] = 0
    return x


def loss_calculate(y_hat, y):
    return -np.log(y_hat[int(y)])


# Forward the input instance through the network
def forward_propagation(picked_example_x, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    picked_example_x = np.reshape(picked_example_x, (-1, 1))
    z1 = np.dot(w1, picked_example_x) + b1
    h1 = ReLU_for_vector(z1)

    # Norm for reducing the values of z1. Large values cause overflow in the softmax function
    if h1.max() != 0.0:
        if not np.isnan(h1.max()):
            h1 = h1 / h1.max()

    z2 = np.dot(w2, h1) + b2
    y_hat = softmax(z2)
    ret = {'z1': z1, 'h1': h1, 'z2': z2, 'y_hat': y_hat}
    # Turns 2 dictionaries into one dictionary
    for key in params:
        ret[key] = params[key]
    return ret


# Compute the gradients w.r.t all the parameters (backpropagation) and update the parameters using GD/SGD
def back_propagation(picked_example_x, picked_example_y, params, eta):
    w1, b1, w2, b2, z1, h1, y_hat = [params[key] for key in ('w1', 'b1', 'w2', 'b2', 'z1', 'h1', 'y_hat')]
    picked_example_x = np.reshape(picked_example_x, (1, 784))
    y_hat[int(picked_example_y)] -= 1

    gw2 = np.dot(y_hat, np.transpose(h1))
    gb2 = y_hat

    w2 = w2 - (eta * gw2)
    b2 = b2 - (eta * gb2)

    dz1 = np.dot(np.transpose(w2), y_hat) * derivative_ReLU_for_vector(z1)

    gw1 = np.dot(dz1, picked_example_x)
    gb1 = dz1

    w1 = w1 - (eta * gw1)
    b1 = b1 - (eta * gb1)

    ret = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return ret

# Loop over the training set #EPOCH number of times
def training_loop(train_x, train_y, i_params, eta, epochs):
    for i in range(epochs):
        # Shuffle the examples
        np.random.seed(2)
        even_spaced_values = np.arange(len(train_x))
        np.random.shuffle(even_spaced_values)
        train_x = train_x[even_spaced_values]
        train_y = train_y[even_spaced_values]

#        loss = 0
        for picked_example_x, picked_example_y in zip(train_x, train_y):
            # Forward the input instance through the network
            f_params = forward_propagation(picked_example_x, i_params)

            # Calculate the loss - I calculated the loss just to keep the pseudo-code's order in the practice presentation.
            # Since I use backpropagation, I don't really need the loss, so I put it in note
#            loss += loss_calculate(f_params['y_hat'], picked_example_y)

            # Compute the gradients w.r.t all the parameters (backpropagation) and update the parameters using GD/SGD
            b_params = back_propagation(picked_example_x, picked_example_y, f_params, eta)

            for key in i_params:
                i_params[key] = b_params[key]
#        print(loss)
    return i_params


# classifying the examples by the params that returns the maximum class
def classify(test_x, params):
    classifications = []
    for x in test_x:
        classifications.append(np.argmax(forward_propagation(x, params)['y_hat']))
    return classifications


# Prepare data for training
def prepare_data(percent, train_x, train_y):
    num_of_examples = len(train_x)
    num_of_examples_for_learning = int(num_of_examples * percent)
    train_x_for_learn = train_x[:num_of_examples_for_learning]
    train_y_for_learn = train_y[:num_of_examples_for_learning]
    train_x_for_test = train_x[num_of_examples_for_learning:]
    train_y_for_test = train_y[num_of_examples_for_learning:]
    return [train_x_for_learn, train_y_for_learn, train_x_for_test, train_y_for_test]


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


# write classifications to file
def write_classifications_to_file(classifications):
    output_file = open("test_y", "w")
    for c in classifications:
        output_file.write(str(c) + '\n')
    output_file.close()


def check_values_for_hyper_params(train_x, train_y):
    number_of_pixels = 28*28
    number_of_classes = 10
    learn_percent = [0.9, 0.8, 0.7, 0.6]
    epochs_values = [10, 15, 20, 25]
    eta_values = [0.1, 0.01, 0.001, 0.0001]
    hidden_layer_size_values = [160, 140, 120, 100]
    for i in learn_percent:
        for j in epochs_values:
            for k in eta_values:
                for n in hidden_layer_size_values:
                    train_x_for_learn, train_y_for_learn, train_x_for_test, train_y_for_test = prepare_data(i, train_x, train_y)
                    hidden_layer_size = n
                    w1 = np.random.uniform(-0.8, 0.8, (hidden_layer_size, number_of_pixels))
                    b1 = np.random.uniform(-0.8, 0.8, hidden_layer_size)
                    w2 = np.random.uniform(-0.8, 0.8, (number_of_classes, hidden_layer_size))
                    b2 = np.random.uniform(-0.8, 0.8, number_of_classes)
                    i_params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
                    print("learn_percent: " + str(i) + " " + "epochs_values: " + str(j) + " " + "eta_values: " + str(k) + " " + "hidden_layer_size_values: " + str(n))
                    final_params = training_loop(train_x_for_learn, train_y_for_learn, i_params, k, j)
                    classifications = classify(train_x_for_test, final_params)
                    print("error percentages: " + str(count_error_percentages(classifications, train_y_for_test)))


if len(sys.argv) < 4:
        exit(-1)

# Get the arguments
train_x = np.loadtxt(sys.argv[1], delimiter=' ')
train_y = np.loadtxt(sys.argv[2], delimiter=' ')
test_x = np.loadtxt(sys.argv[3], delimiter=' ')

#check_values_for_hyper_params(train_x, train_y)

# Norm
train_x = z_score_norm(train_x)
test_x = z_score_norm(test_x)

# Prepare data for training
train_x_for_learn, train_y_for_learn, train_x_for_test, train_y_for_test = prepare_data(0.9, train_x, train_y)

# Init w1, b1, w2, b2
w1 = np.random.uniform(-0.8, 0.8, [HIDDEN_LAYER_SIZE, NUMBER_OF_PIXELS])
b1 = np.random.uniform(-0.8, 0.8, [HIDDEN_LAYER_SIZE, 1])
w2 = np.random.uniform(-0.8, 0.8, [NUMBER_OF_CLASSES, HIDDEN_LAYER_SIZE])
b2 = np.random.uniform(-0.8, 0.8, [NUMBER_OF_CLASSES, 1])
i_params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

# Init the hyper parameters
eta = 0.01
epochs = 15

# Training
final_params = training_loop(train_x_for_learn, train_y_for_learn, i_params, eta, epochs)

classifications = classify(train_x_for_test, final_params)
print(count_error_percentages(classifications, train_y_for_test))

# classify test_x
classifications = classify(test_x, final_params)
write_classifications_to_file(classifications)
