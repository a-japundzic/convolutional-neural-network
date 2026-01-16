import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from VGG11 import VGG11

# Method to train the model (reformatted it this way to make parts c and d easier)
def train_model(train_loader, model, device, loss_function, optimizer):
    # Train the model (following the training loop and per epoch activity from the pytorch documentation)
    # Set the model to training mode and train one epoch
    model.train(True)

    # Total loss for the epoch
    running_loss = 0
    # Total number of labels in the epoch
    total_labels = 0
    # Total number of correct predictions in the epoch
    correct = 0
    for (images, labels) in train_loader:
        # Move the images and labels to the GPU
        images = images.to(device)
        labels = labels.to(device)

        # zero the gradients
        optimizer.zero_grad()

        # Make predictions
        outputs = model(images)

        # Calculate the loss and its gradients (as discussed in class)
        loss = loss_function(outputs, labels)
        loss.backward()

        # Adjust the learning rate
        optimizer.step()

        # Track statistics
        # Have to multiply by the batch size to get the total loss for the batch
        running_loss += loss.item() * images.size(0)
        # Predict the class with the highest probability (10 classes for MINST, each correspondding to a number 0-9)
        predicted = torch.argmax(outputs, dim=1)
        # Add up the total number of labels in the batch
        total_labels += labels.size(0)
        # Add up the total number of correct predictions in the batch
        correct += (predicted == labels).sum().item()

    # Calculate the average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct / total_labels
    print(f"  Train Loss: {epoch_loss:.4f} || Train Accuracy: {epoch_accuracy:.4f}")


    return epoch_loss, epoch_accuracy

# Method to test the model (reformatted it this way to make parts c and d easier)
def test_model(test_loader, model, device, loss_function):
    # Set the model to evaluation mode to test
    model.eval()

    # Total loss for the epoch
    running_loss = 0
    # Total number of labels in the epoch
    total_labels = 0
    # Total number of correct predictions in the epoch
    correct = 0
    # Disable the gradient calculation as we are no longer optimizing
    # (i.e. no need for backpropagation)
    # This should speed up the testing process
    with torch.no_grad():
        for (images, labels) in test_loader:
            # Move the images and labels to the GPU (or to CPU if no GPU is available)
            images = images.to(device)
            labels = labels.to(device)
            
            # Make predictions
            outputs = model(images)

            # Calculate the loss (no need for gradients as we are not optimizing)
            loss = loss_function(outputs, labels)
            # Again have to multiply by the batch size to get the total loss for the batch
            running_loss += loss.item() * images.size(0)
            # Predict the class with the highest probability
            predicted = torch.argmax(outputs, dim=1)
            # Add up the toal number of labels in the batch
            total_labels += labels.size(0)
            # Add up the toal number of correct predictions in the batch
            correct += (predicted == labels).sum().item()

    # Calculate the average loss and accuracy for the epoch
    epoch_test_loss = running_loss / len(test_loader.dataset)
    epoch_test_acc = correct / total_labels
    print(f"  Test Loss: {epoch_test_loss:.4f}  || Test Accuracy: {epoch_test_acc:.4f}")

    return epoch_test_loss, epoch_test_acc


if __name__ == "__main__":
    # Variable to hold the number of epochs (in this case 5)
    EPOCH = 5
    
    # a)
    print("Basic: No generalization or regularization")
    # Transform the data
    transform_a = transforms.Compose([
        # 32 by 32 as mentioned in the assignment
        transforms.Resize((32, 32)),
        # Convert the image to a tensor (as required by PyTorch)
        transforms.ToTensor(),
        # Following what the paper does and subtracting the mean pixel value
        # I searched up the MNIST mean and found that it is 0.1307
        transforms.Normalize(mean=[0.1307], std=[1.0])
    ])

    # Load the data (shuffle the training data for each epoch)
    train_dataset_a = datasets.MNIST(root='./data', train=True, download=True, transform=transform_a)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_a)

    # Wrap in DataLoader
    # Set batch size to 256 (followed what the paper did)
    train_loader_a = DataLoader(train_dataset_a, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Setup the model so that it is on the GPU (im lucky enough to have a 3080 for this assignment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG11().to(device)

    # Setup the loss function (assignment said to use cross entropy loss)
    loss_function = nn.CrossEntropyLoss()
    # Setup the optimizer (followed what the paper did)
    optimizer = optim.SGD(model.parameters(), weight_decay=5e-4, lr=0.01, momentum=0.9)

    # Lists to store metrics for plotting
    train_losses_a = []
    train_accuracies_a = []
    test_losses_a = []
    test_accuracies_a = []

    # Train and test model for each epoch
    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}:")
        # Train and test!
        epoch_train_loss, epoch_train_acc = train_model(train_loader_a, model, device, loss_function, optimizer)
        epoch_test_loss, epoch_test_acc = test_model(test_loader, model, device, loss_function)

        # Append the metrics to the lists for plotting
        train_losses_a.append(epoch_train_loss)
        train_accuracies_a.append(epoch_train_acc)
        test_losses_a.append(epoch_test_loss)
        test_accuracies_a.append(epoch_test_acc)

    # b)
    # Plot
    # Spent some time producing this cool graph, hope you enjoy it!
    epochs = range(1, EPOCH + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    
    axs[0, 0].plot(epochs, test_accuracies_a, 'b-o', label='Test Acc')
    axs[0, 0].set_title('Test Accuracy (higher is better)')
    axs[0, 0].set_ylabel('Accuracy (%)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, train_accuracies_a, 'g-o', label='Train Acc')
    axs[0, 1].set_title('Training Accuracy (higher is better)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, test_losses_a, 'r-o', label='Test Loss')
    axs[1, 0].set_title('Test Loss (lower is better)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(epochs, train_losses_a, 'm-o', label='Train Loss')
    axs[1, 1].set_title('Training Loss (lower is better)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


    # c)
    print("Generalization of test set")

    # Horizontal flip
    transform_b_hf = transforms.Compose([
        # 32 by 32 as mentioned in the assignment
        transforms.Resize((32, 32)),
        # Flip the image horizontally (before tensor/normalize)
        transforms.RandomHorizontalFlip(p=1),
        # Convert the image to a tensor (as required by PyTorch)
        transforms.ToTensor(),
        # Following what the paper does and subtracting the mean pixel value
        # I searched up the MNIST mean and found that it is 0.1307
        transforms.Normalize(mean=[0.1307], std=[1.0]),
    ])

    # Use the test dataset as required by the assignment
    test_dataset_b_hf = datasets.MNIST(root='./data', train=False, download=True, transform=transform_b_hf)
    test_loader_b_hf = DataLoader(test_dataset_b_hf, batch_size=256, shuffle=False)
 
    print("Test Accuracy with horizontal flip:")
    test_model(test_loader_b_hf, model, device, loss_function)

    # Vertical flip
    transform_b_vf = transforms.Compose([
        # 32 by 32 as mentioned in the assignment
        transforms.Resize((32, 32)),
        # Flip vertically (apply before tensor/normalize)
        transforms.RandomVerticalFlip(p=1),
        # Convert the image to a tensor (as required by PyTorch)
        transforms.ToTensor(),
        # Following what the paper does and subtracting the mean pixel value
        # I searched up the MNIST mean and found that it is 0.1307
        transforms.Normalize(mean=[0.1307], std=[1.0]),
    ])

    test_dataset_b_vf = datasets.MNIST(root='./data', train=False, download=True, transform=transform_b_vf)
    test_loader_b_vf = DataLoader(test_dataset_b_vf, batch_size=256, shuffle=False)

    print("Test Accuracy with vertical flip:")
    test_model(test_loader_b_vf, model, device, loss_function)


    # Gaussian noise
    variance = [0.01, 0.1, 1]
    for var in variance:
        transform_b_gn = transforms.Compose([
            # 32 by 32 as mentioned in the assignment
            transforms.Resize((32, 32)),
            # Convert the image to a tensor (as required by PyTorch)
            transforms.ToTensor(),
            # Add gaussian noise with specified variance
            transforms.Lambda(lambda x: x + var * torch.randn_like(x)),
            # Following what the paper does and subtracting the mean pixel value
            # I searched up the MNIST mean and found that it is 0.1307
            transforms.Normalize(mean=[0.1307], std=[1.0]),
        ])

        test_dataset_b_gn = datasets.MNIST(root='./data', train=False, download=True, transform=transform_b_gn)
        test_loader_b_gn = DataLoader(test_dataset_b_gn, batch_size=256, shuffle=False)

        print(f"Test Accuracy with Gaussian noise (variance: {var}):")
        test_model(test_loader_b_gn, model, device, loss_function)


    #d)
    print("Retraining with data augmentation")

    # Random horiztonal flip and random rotation
    # Chose these augmentations as I feel these are the most common way for numbers to show up in practice
    # I know when I play UNO, it is hard to distinguies between 6 and 9, maybe I should train a model to help me
    # Also incluided Gaussian noise as when I reran with the test from part c I noticed that the model was not
    #  performing well on the blurred numbers
    transform_d = transforms.Compose([
        # 32 by 32 as mentioned in the assignment
        transforms.Resize((32, 32)),
        # Flip the image horizontally (before tensor/normalize)
        transforms.RandomHorizontalFlip(p=0.5),
        # Rotate the image randomly (before tensor/normalize)
        transforms.RandomRotation(degrees=180),
        # Convert the image to a tensor (as required by PyTorch)
        transforms.ToTensor(),
        # Add Gaussian noise 
        # Uniformly draw variance between 0 and 1
        transforms.Lambda(lambda x: x + random.uniform(0,1) * torch.randn_like(x)),
        # Following what the paper does and subtracting the mean pixel value
        # I searched up the MNIST mean and found that it is 0.1307)
        transforms.Normalize(mean=[0.1307], std=[1.0]),
    ])

    # Reinitialize the model and optimizer (I don't know if this is necessary but included it anyways)
    model_d = VGG11().to(device)
    optimizer_d = optim.SGD(model_d.parameters(), weight_decay=5e-4, lr=0.01, momentum=0.9)

    # Load the data with the data augmentation defined above
    train_dataset_d = datasets.MNIST(root='./data', train=True, download=True, transform=transform_d)
    train_loader_d = DataLoader(train_dataset_d, batch_size=256, shuffle=True)

    # Train and test the model on the original data
    # With a GPU I had the luxary of using 5 epochs here as well
    print("Train accuracy and test accuracy with data augmentation on original data:")
    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}:")
        train_model(train_loader_d, model_d, device, loss_function, optimizer_d)
        test_model(test_loader, model_d, device, loss_function)

    print("Test accuracy with horizontal flip:")
    test_model(test_loader_b_hf, model_d, device, loss_function)

    print("Test accuracy with vertical flip:")
    test_model(test_loader_b_vf, model_d, device, loss_function)

    # Gaussian noise (as in part c)
    # I lazially just copied and pasted this again as I didn't want to format it into its own function
    for var in variance:
        transform_b_gn = transforms.Compose([
            # 32 by 32 as mentioned in the assignment
            transforms.Resize((32, 32)),
            # Convert the image to a tensor (as required by PyTorch)
            transforms.ToTensor(),
            # Add gaussian noise with specified variance
            transforms.Lambda(lambda x: x + var * torch.randn_like(x)),
            # Following what the paper does and subtracting the mean pixel value
            # I searched up the MNIST mean and found that it is 0.1307
            transforms.Normalize(mean=[0.1307], std=[1.0]),
        ])

        # Use the test split as required by the assignment
        test_dataset_b_gn = datasets.MNIST(root='./data', train=False, download=True, transform=transform_b_gn)
        test_loader_b_gn = DataLoader(test_dataset_b_gn, batch_size=256, shuffle=False)

        print(f"Test Accuracy with Gaussian noise (variance: {var}):")
        test_model(test_loader_b_gn, model_d, device, loss_function)
