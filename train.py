from models.lstm import LSTM_NN
from utils.data.motion_dataset import FilteredMotionDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import datetime

# Create the necessary dataloaders
#
# Training
training_dataset = FilteredMotionDataset(
    "./data/uncompressed/tf_example/training/")
training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
# Validation
validation_dataset = FilteredMotionDataset(
    "./data/uncompressed/tf_example/validation/")
validation_dataloader = DataLoader(
    validation_dataset, batch_size=32, shuffle=False)
# Testing
test_dataset = FilteredMotionDataset("./data/uncompressed/tf_example/testing/")
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Setup necessary input sizes for the model
agent_input, static_roadgraph_input, dynamic_roadgraph_input, agent_target, agent_target_states_valid = training_dataset[
    0]

num_agent_features = agent_input.size(dim=-1)
num_static_roadgraph_features = static_roadgraph_input.size(dim=-1)
num_dynamic_roadgraph_features = dynamic_roadgraph_input.size(dim=-1)
num_future_features = agent_target.size(dim=-1)
num_future_timesteps = agent_target.size(dim=-2)
agent_hidden_size = 32
static_roadgraph_hidden_size = 64
dynamic_roadgraph_hidden_size = 32
# Setup model inputs and outputs
model = LSTM_NN(num_agent_features=num_agent_features,
                num_static_road_features=num_static_roadgraph_features,
                num_dynamic_road_features=num_dynamic_roadgraph_features,
                agent_hidden_size=agent_hidden_size,
                static_roadgraph_hidden_size=static_roadgraph_hidden_size,
                dynamic_roadgraph_hidden_size=dynamic_roadgraph_hidden_size,
                num_future_features=num_future_features,
                num_future_timesteps=num_future_timesteps)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 1
for epoch in range(epochs):
    # Training
    model.train()  # Set model to training mode
    train_loss = 0.0
    for dataset_element in training_dataloader:
        agent_input, static_roadgraph_input, dynamic_roadgraph_input, agent_target, agent_target_states_valid = dataset_element
        optimizer.zero_grad()
        outputs = model(agent_input, static_roadgraph_input,
                        dynamic_roadgraph_input)
        # Mask the outputs where the agent_target_states_valid is 0
        outputs = torch.where(
            agent_target_states_valid.unsqueeze(dim=-1) == 0,
            torch.ones_like(agent_target),
            outputs,
        )
        loss = criterion(outputs, agent_target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(training_dataloader)

    # Validation
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for dataset_element in validation_dataloader:
            agent_input, static_roadgraph_input, dynamic_roadgraph_input, agent_target, agent_target_states_valid = dataset_element
            optimizer.zero_grad()
            outputs = model(agent_input, static_roadgraph_input,
                            dynamic_roadgraph_input)
            # Mask the outputs where the agent_target_states_valid is 0
            outputs = torch.where(
                agent_target_states_valid.unsqueeze(dim=-1) == 0,
                torch.ones_like(agent_target),
                outputs,
            )
            loss = criterion(outputs, agent_target)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(validation_dataloader)

    print(
        f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

now = datetime.datetime.now()
path = "./models/trained_weights/lstm_model_" + \
    now.strftime("%Y-%m-%d %H:%M:%S") + ".pt"
torch.save(model.state_dict(), path)
