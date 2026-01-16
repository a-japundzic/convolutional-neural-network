# convolutional-neural-network
VGG-11 Neural Network for classifying MNIST dataset

A Convolutional Neural Network (CNN) is a neural network designed to process grid-like data, especially images. It learns spatially local patterns (like edges and textures) and combines them into higher-level features (like shapes and objects).

## a)
Results:

<img width="1194" height="899" alt="image" src="https://github.com/user-attachments/assets/fa3fb928-3c1b-46fb-9463-4facadc83bb1" />

## b)

```jsx
// Test Accuracy with horizontal flip:
  Test Loss: 5.1716  || Test Accuracy: 0.3851
// Test Accuracy with vertical flip:
  Test Loss: 4.4607  || Test Accuracy: 0.4528
// Test Accuracy with Gaussian noise (variance: 0.01):
  Test Loss: 0.0177  || Test Accuracy: 0.9940
// Test Accuracy with Gaussian noise (variance: 0.1):
  Test Loss: 0.0819  || Test Accuracy: 0.9739
// Test Accuracy with Gaussian noise (variance: 1):
  Test Loss: 17.2347  || Test Accuracy: 0.0988
```

Here we see that both flipping vertically and horizontally greatly effect the test accuracy. This is because digits are not symmetric and flipping them makes it hard for even a human to recognize them.

We also see when blurring the image, that with a slight blur (variance 0.01, and 0.1) the numbers are still recognizable enough that the model is able to get high test accuracy. Eventually when the noise becomes too much (variance 1), the accuracy plummets to a shockingly low number. This tells us that the model is currently sensitive to lots of noise in the image, meaning blurry images perform worse than clear images. 

## c)

I wanted to address the three issues identified in part b) with my model. To do this, I applied the following data augmentation:

```python
  # Flip the image horizontally (before tensor/normalize)
  transforms.RandomHorizontalFlip(p=0.5)
  # Rotate the image randomly (before tensor/normalize)
  transforms.RandomRotation(degrees=180)
  # Add Gaussian noise 
  # Uniformly draw variance between 0 and 1
  transforms.Lambda(lambda x: x + random.uniform(0,1) * torch.randn_like(x))
```

With random horizontal flips, the model gets to train on both regular numbers and horizontally flipped numbers. I combined this with random rotation to put the numbers in all sorts of different orders. My thought process was related to the game of Uno. No matter where you are at the table, we as humans are usually able to recognize the numbers (provided they are written with some clarity) no matter their orientation. I applied these augmentations to account for that.

Finally, I applied a random amount of noise to each image. my goal here was to make the model better at classifying images of numbers that have a lot of blur. 

When rerunning using the generalizations from part b), I got the following results this time:

```jsx
// Test accuracy with horizontal flip:
  Test Loss: 0.2343  || Test Accuracy: 0.9210
// Test accuracy with vertical flip:
  Test Loss: 0.2383  || Test Accuracy: 0.9198
// Test Accuracy with Gaussian noise (variance: 0.01):
  Test Loss: 0.2002  || Test Accuracy: 0.9326
// Test Accuracy with Gaussian noise (variance: 0.1):
  Test Loss: 0.2021  || Test Accuracy: 0.9322
// Test Accuracy with Gaussian noise (variance: 1):
  Test Loss: 0.9216  || Test Accuracy: 0.6944
```

Where I saw improvements  in every result. The most impressive to me is the flips and the gaussian noise with variance 1 getting an accuracy boost of 70 percent! Overall, the accuracy of the model went down a bit in some areas, but in “general” it is much better at classifying numbers.
