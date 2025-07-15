import torch
import torch.nn as nn
import torch.optim as optim

class BinarySelectionNet(nn.Module):
    def __init__(self, n_features):
        super(BinarySelectionNet, self).__init__()
        # Define a simple architecture for demonstration purposes.
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, n_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output logits before applying the STE
        return x

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass the gradient unchanged
        return grad_output

ste_binarize = STEFunction.apply

def train_model(model, inputs, labels, epochs=100, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()  # Use logits directly
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        binarized_outputs = ste_binarize(outputs.sigmoid())  # Apply STE during training
        outputs = binarized_outputs*inputs
        loss = criterion(binarized_outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Assuming you have your input data `inputs` and corresponding binary labels `labels`
n_features = 10  # Example number of features
model = BinarySelectionNet(n_features)

# Create dummy data for demonstration
inputs = torch.rand((5, n_features))  # Random input vectors
labels = (torch.rand((5, n_features)) > 0.5).float()  # Random binary labels

# Train the model with the dummy data
train_model(model, inputs, labels)

# After training, you can use the model to predict vj from vi
model.eval()
with torch.no_grad():
    vi = torch.rand((1, n_features))  # Example single input vector
    logits = model(vi)
    vj = ste_binarize(logits.sigmoid())  # Get binary output using STE
    vij = vi * vj  # Element-wise multiplication

print("Input vector vi:", vi)
print("Binary selection vector vj:", vj)
print("Filtered vector vij:", vij)