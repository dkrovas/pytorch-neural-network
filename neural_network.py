import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if training:
        train_set=datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        
    else:
        print("False")
        train_set=datasets.FashionMNIST('./data', train=False, transform=transform)
    loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    return loader



def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64,10)
    )
    return model
    
def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(0,T):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for pair in train_loader:
            inputs, labels = pair
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = 100*(correct_predictions/total_samples)
        running_loss = running_loss/total_samples
        print(f"Train Epoch: {epoch} Accuracy: {correct_predictions}/{total_samples}({accuracy:.2f}%) Loss: {running_loss:.3f}")


def evaluate_model(model, train_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for pair in train_loader:
            inputs, labels = pair
           
            outputs = model(inputs)
            loss = criterion(outputs,labels)

            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    if(show_loss):
        print(f"Average loss: {running_loss/total_samples:.4f}")

    print(f"Accuracy: {100*(correct_predictions/total_samples):.2f}%")
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt'
,'Sneaker','Bag','Ankle Boot']
    image = test_images[index]

    logits = model(image.unsqueeze(0))
    
    
    prob = F.softmax(logits, dim=1)

    top3_prob, top3_index = torch.topk(prob, 3)

    for i in range(3):
        class_index = top3_index[0][i]
        probability = top3_prob[0][i]*100
        class_name = class_names[class_index]
        print(f"{class_name}: {probability:.2f}%")
    

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    model = build_model()
    print(model)
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model.eval(), train_loader, criterion, show_loss=True)
    evaluate_model(model.eval(), train_loader, criterion, show_loss=False)
    test_images = next(iter(train_loader))[0]
    predict_label(model, test_images, 1)
