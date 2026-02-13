# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset
Develop a Fashion Item Classification System using a Convolutional Neural Network (CNN) to classify images from the Fashion-MNIST dataset. The model should accurately categorize grayscale images of clothing items into one of 10 classes, such as T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots.
# Dataset
![image](https://github.com/user-attachments/assets/4df6dfa4-f5ae-430d-b0b5-7a47c3c274e2)


## Neural Network Model
![Screenshot 2025-03-22 213528](https://github.com/user-attachments/assets/64a96d4d-e066-4d45-aa5d-4096768ec69d)



## DESIGN STEPS

### STEP 1:
Import required libraries.Then dataset is loaded and define the training and testing dataset.

### STEP 2:
Define the CNN Model consist of three convolutional layers with ReLU activation,MaxPooling layers to reduce spatial dimensions,fully connected layers for final classification.

### STEP 3:
initialize the model,loss function,optimizer. CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

### STEP 4:
Train the model with training dataset.

### STEP 5:
Evaluate the model with testing dataset.

### STEP 6:
Make Predictions on New Data.

## PROGRAM

### Name: PARTHASARATHI S
### Register Number:212223040144
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn. Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn. MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn. Linear (128 * 3 * 3, 128)
        self.fc2 = nn. Linear (128, 64)
        self.fc3= nn. Linear (64, 10)
    def forward(self, x):
        x = self.pool (torch.relu(self.conv1(x)))
        x = self.pool (torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size (0), -1) # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT

### Training Loss per Epoch
![image](https://github.com/user-attachments/assets/1d803669-8b93-40ea-8ba3-d493ac14e604)


### Confusion Matrix

![image](https://github.com/user-attachments/assets/dafef555-be60-4fec-a13e-dc54f6bb67d0)


### Classification Report
![image](https://github.com/user-attachments/assets/6e1f2f4d-dfc0-447b-b4a9-bbe08ba62fb5)



### New Sample Data Prediction
![Screenshot 2025-03-22 215611](https://github.com/user-attachments/assets/43ac055c-5ade-4597-b920-fcbc545230db)

![image](https://github.com/user-attachments/assets/b65e25df-625d-4556-acb8-6dd18c9bad8d)


## RESULT
Thus, the convolutional deep neural network for image classification and to verify the response for new images has developed succesfully.
