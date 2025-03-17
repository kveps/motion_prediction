from models.lstm import LSTM_NN
from utils.data.motion_dataset import FilteredMotionDataset
from utils.viz.visualize_scenario import visualize_model_inputs_and_output
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
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Setup necessary input sizes for the model
dummy_element = training_dataset[0]
agent_input = dummy_element['agent_input']
static_roadgraph_input = dummy_element['static_roadgraph_input']
dynamic_roadgraph_input = dummy_element['dynamic_roadgraph_input']
agent_target = dummy_element['agent_target']
agent_target_valid = dummy_element['agent_target_valid']

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

TRAINING_MODE = False

if TRAINING_MODE:
    # Training Loop
    epochs = 100
    for epoch in range(epochs):
        # Training
        model.train()  # Set model to training mode
        train_loss = 0.0
        for dataset_element in training_dataloader:
            # fetch inputs
            agent_input = dataset_element['agent_input']
            static_roadgraph_input = dataset_element['static_roadgraph_input']
            dynamic_roadgraph_input = dataset_element['dynamic_roadgraph_input']
            agent_target = dataset_element['agent_target']
            agent_target_valid = dataset_element['agent_target_valid']

            optimizer.zero_grad()
            outputs = model(agent_input, static_roadgraph_input,
                            dynamic_roadgraph_input)
            # Mask the outputs where the agent_target_valid is 0
            outputs = torch.where(
                agent_target_valid.unsqueeze(dim=-1) == 0,
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
                # fetch inputs
                agent_input = dataset_element['agent_input']
                static_roadgraph_input = dataset_element['static_roadgraph_input']
                dynamic_roadgraph_input = dataset_element['dynamic_roadgraph_input']
                agent_target = dataset_element['agent_target']
                agent_target_valid = dataset_element['agent_target_valid']

                optimizer.zero_grad()
                outputs = model(agent_input, static_roadgraph_input,
                                dynamic_roadgraph_input)
                # Mask the outputs where the agent_target_valid is 0
                outputs = torch.where(
                    agent_target_valid.unsqueeze(dim=-1) == 0,
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
            str(epoch + 1) + "_" + now.strftime("%Y-%m-%d %H:%M:%S") + ".pt"
        torch.save(model.state_dict(), path)
else:
    # Testing
    model_path = "./models/trained_weights/lstm_model_17_2025-03-16 13:32:17.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():
        for dataset_element in test_dataloader:
            # fetch inputs
            agent_input = dataset_element['agent_input']
            static_roadgraph_input = dataset_element['static_roadgraph_input']
            dynamic_roadgraph_input = dataset_element['dynamic_roadgraph_input']
            agent_target = dataset_element['agent_target']
            agent_target_valid = dataset_element['agent_target_valid']
            tracks_to_predict = dataset_element['tracks_to_predict']

            optimizer.zero_grad()
            outputs = model(agent_input, static_roadgraph_input,
                            dynamic_roadgraph_input)
            loss = criterion(outputs, agent_target)
            test_loss += loss.item()

            # Visualize the model inputs and outputs
            model_output = {
                'agent_output': outputs
            }
            visualize_model_inputs_and_output(
                dataset_element, model_output)

    avg_test_loss = test_loss / len(test_dataloader)

    print(f'Test Loss: {avg_test_loss:.4f}')
