# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:
Model Design:

Input Layer: Number of neurons = features.

Hidden Layers: 2 layers with ReLU activation.

Output Layer: 4 neurons (segments A, B, C, D) with softmax activation.

### STEP 3:

Model Compilation: Use categorical crossentropy loss, Adam optimizer, and track accuracy.

### STEP 4:

Training: Train with early stopping, batch size (e.g., 32), and suitable epochs.


### STEP 5:

Evaluation: Assess using accuracy, confusion matrix, precision, and recall.


### STEP 6:

Optimization: Tune hyperparameters (layers, neurons, learning rate, batch size).


## PROGRAM

### Name: A.LAHARI
### Register Number: 212223230111

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,30)
        self.fc2 = nn.Linear(30,14)
        self.fc3 = nn.Linear(14,6)
        self.fc4 = nn.Linear(6,4)

    def forward(self,x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

```
```python
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

```
```python
def train_model(model, train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      output = model(X_batch)
      loss = criterion(output,y_batch)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```



## Dataset Information

<img width="1028" height="664" alt="image" src="https://github.com/user-attachments/assets/c68c804b-d2eb-4d8b-a515-2e4d83bf7516" />

## OUTPUT

### Confusion Matrix

<img width="685" height="567" alt="image" src="https://github.com/user-attachments/assets/52978871-f027-4773-b029-740d34d2f858" />



### Classification Report

<img width="1390" height="582" alt="image" src="https://github.com/user-attachments/assets/180e9584-95a1-44fe-a19d-9d3c71563cf2" />



### New Sample Data Prediction

<img width="373" height="105" alt="image" src="https://github.com/user-attachments/assets/5a6ff0ae-8567-4fdd-b87c-86e072158c42" />


## RESULT
Thus a neural network classification model for the given dataset is executed successfully.

