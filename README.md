# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model


<img width="798" height="737" alt="image" src="https://github.com/user-attachments/assets/82afc512-c560-4a7a-9771-bc26c913d0fb" />


## DESIGN STEPS

### STEP 1:

Import the required libraries for data handling and neural networks.

### STEP 2:

Load the dataset and explore its structure.

### STEP 3:

Clean the dataset and handle missing values if present.

### STEP 4:

Encode categorical variables into numerical format.

### STEP 5:

Normalize or scale the numerical features.


### STEP 6:

Split the dataset into training and testing sets.

### STEP 7:

Define the neural network architecture .

### STEP 8:

Select CrossEntropyLoss as the loss function and Adam as the optimizer.

### STEP 9:
Train the model using forward pass, loss calculation, backpropagation, and weight updates.

### STEP 10:
Evaluate the model using accuracy, confusion matrix, and classification report.

## PROGRAM

### Name: A.LAHARI
### Register Number: 212223230111

```python

class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 20)
        self.fc3 = nn.Linear(20, 16)
        self.fc4 = nn.Linear(16, 4)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
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

<img width="664" height="559" alt="image" src="https://github.com/user-attachments/assets/9f3fff5c-7b1b-4f28-9832-87e5891522d2" />


### Classification Report

<img width="540" height="436" alt="image" src="https://github.com/user-attachments/assets/754410b3-3447-4238-8192-1096c3c77b12" />



### New Sample Data Prediction

<img width="351" height="103" alt="image" src="https://github.com/user-attachments/assets/ad50184f-c561-476a-91af-72e5d73b7a42" />




## RESULT
Thus a neural network classification model for the given dataset is executed successfully.

