import numpy as np
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms


# load train data and test data and split the train to train and validation
def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0],[1])])
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

    # split the training set to train and validation (80:20)
    num_of_examples = len(train_dataset)
    num_of_examples_for_learning = int(num_of_examples * 0.8)
    num_of_examples_for_testing = int(num_of_examples - num_of_examples_for_learning)

    train_dataset, validation_dataset = random_split(train_dataset, [num_of_examples_for_learning, num_of_examples_for_testing])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, test_loader


# model A implement
class ModelA (nn.Module):
    def __init__(self,image_size):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# model B implement
class ModelB (nn.Module):
    def __init__(self,image_size):
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# model C implement
class ModelC (nn.Module):
    def __init__(self,image_size):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.batch1 = nn.BatchNorm1d(100)
        self.batch2 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.batch1(self.fc0(x))
        x = F.relu(x)
        x = self.batch2(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# model D implement
class ModelD (nn.Module):
    def __init__(self,image_size):
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


# model E implement
class ModelE (nn.Module):
    def __init__(self,image_size):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


# train
def train(model, optimizer, train_loader):
    model.train()
    average_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        average_loss += F.nll_loss(output, labels, reduction='sum').data # sum up batch loss
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).cpu().sum()

    average_loss /= len(train_loader.dataset)
    average_correct = correct / float(len(train_loader.dataset))
    return average_loss, average_correct


# validate
def validate(model, validation_loader):
    model.eval()
    average_loss = 0
    correct = 0
    for data, target in validation_loader:
        output = model(data)
        average_loss += F.nll_loss(output, target, reduction='sum').data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).cpu().sum()

    average_loss /= len(validation_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(average_loss, correct, len(validation_loader.dataset), 100. * correct / len(validation_loader.dataset)))
    average_correct = correct / float(len(validation_loader.dataset))
    return average_loss, average_correct


# train and validate the model
def train_and_validate_model(model, train_loader, validation_loader, epochs, eta):
    train_loss = []
    validation_loss = []
    train_accuracy = []
    validation_accuracy = []

    optimizer = optim.Adam(model.parameters(), lr=eta)

    for i in range(epochs):
        one_epoch_train_loss, one_epoch_train_accuracy = train(model, optimizer, train_loader)
        train_loss.append(one_epoch_train_loss.item())
        train_accuracy.append(one_epoch_train_accuracy.item())
        one_epoch_validation_loss, one_epoch_validation_accuracy = validate(model, validation_loader)
        validation_loss.append(one_epoch_validation_loss.item())
        validation_accuracy.append(one_epoch_validation_accuracy.item())

    return train_loss, validation_loss, train_accuracy, validation_accuracy

# plot the value per epoch for the validation and training set in a single image
def plot(train_value, validation_value, title, value):
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(value)
    plt.xlim(1, 10)

    plt.minorticks_on()
    plt.tick_params(direction='out', color='black')
    plt.grid(color='black', alpha=0.5, which='both')

    print(train_value)
    print(validation_value)

    plt.plot(list(range(1, len(train_value) + 1)), list(train_value), color='black', linestyle='dotted')
    plt.plot(list(range(1, len(validation_value) + 1)), list(validation_value), color='green', linestyle='dotted')

    plt.legend(('Train', 'validation'), fontsize='small')

    plt.show()


# classify by the model
def classify(model, test_x):
    classifications = []
    model.eval()
    for data in test_x:
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        classifications.append(int(pred))
    return classifications


# write classifications to file
def write_classifications_to_file(classifications):
    output_file = open("test_y", "w")
    for c in classifications:
        output_file.write(str(c) + '\n')
    output_file.close()


image_size = 28*28
epochs = 10
batch_size = 64

train_loader, validation_loader, test_loader = load_data(batch_size)

test_x = np.loadtxt(sys.argv[1], delimiter=' ')
test_x = test_x / 255
test_x = torch.Tensor(test_x)
test_x = test_x.reshape(-1 ,28 ,28)
#torch.set_printoptions(edgeitems=11)
#print(test_x)

# model A
#eta = 0.001
#print("model A")
#model_A = ModelA(image_size)
#train_loss, validation_loss, train_accuracy, validation_accuracy = train_and_validate_model(model_A, train_loader, validation_loader, epochs, eta)
#plot(train_loss, validation_loss, "The average loss per epoch for the validation and training set", "Loss")
#plot(train_accuracy, validation_accuracy, "The average accuracy per epoch for the validation and training set", "Accuracy")

# model B
#eta = 0.001
#print("model B")
#model_B = ModelB(image_size)
#train_loss, validation_loss, train_accuracy, validation_accuracy = train_and_validate_model(model_B, train_loader, validation_loader, epochs, eta)
#plot(train_loss, validation_loss, "The average loss per epoch for the validation and training set", "Loss")
#plot(train_accuracy, validation_accuracy, "The average accuracy per epoch for the validation and training set", "Accuracy")

# model C
#eta = 0.001
#print("model C")
#model_C = ModelC(image_size)
#train_loss, validation_loss, train_accuracy, validation_accuracy = train_and_validate_model(model_C, train_loader, validation_loader, epochs, eta)
#plot(train_loss, validation_loss, "The average loss per epoch for the validation and training set", "Loss")
#plot(train_accuracy, validation_accuracy, "The average accuracy per epoch for the validation and training set", "Accuracy")

# model D
#eta = 0.001
#print("model D")
#model_D = ModelD(image_size)
#train_loss, validation_loss, train_accuracy, validation_accuracy = train_and_validate_model(model_D, train_loader, validation_loader, epochs, eta)
#plot(train_loss, validation_loss, "The average loss per epoch for the validation and training set", "Loss")
#plot(train_accuracy, validation_accuracy, "The average accuracy per epoch for the validation and training set", "Accuracy")

# model E
#eta = 0.01
#print("model E")
#model_E = ModelE(image_size)
#train_loss, validation_loss, train_accuracy, validation_accuracy = train_and_validate_model(model_E, train_loader, validation_loader, epochs, eta)
#plot(train_loss, validation_loss, "The average loss per epoch for the validation and training set", "Loss")
#plot(train_accuracy, validation_accuracy, "The average accuracy per epoch for the validation and training set", "Accuracy")

# best model
eta = 0.001
best_model = ModelC(image_size)
train_loss, validation_loss, train_accuracy, validation_accuracy = train_and_validate_model(best_model, train_loader, validation_loader, epochs, eta)

classifications = classify(best_model, test_x)
write_classifications_to_file(classifications)
