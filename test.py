from models.loss import trajectory_prediction_multi_modal_loss
from models.lstm import LSTM_NN
from utils.data.motion_dataset import FilteredMotionDataset
from utils.viz.visualize_scenario import visualize_model_inputs_and_output
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

# Create the necessary dataloaders
# Testing
test_dataset = FilteredMotionDataset(
    "./data/uncompressed/tf_example/training/")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Setup necessary input sizes for the model
dummy_element = test_dataset[0]
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
num_future_trajectoiries = 4
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
                num_future_trajectories=num_future_trajectoiries,
                num_future_timesteps=num_future_timesteps)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Testing
model_path = "./models/trained_weights/lstm_model_2_2025-03-17 10:00:53.pt"
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
        trajectories, probs = model(agent_input, static_roadgraph_input,
                                    dynamic_roadgraph_input)
        loss = trajectory_prediction_multi_modal_loss(
            trajectories, probs, agent_target, agent_target_valid)
        test_loss += loss.item()

        # Visualize the model inputs and outputs
        model_output = {
            'agent_trajs': trajectories,
            'agent_probs': probs,
        }
        visualize_model_inputs_and_output(
            dataset_element, model_output)

avg_test_loss = test_loss / len(test_dataloader)

print(f'Test Loss: {avg_test_loss:.4f}')
