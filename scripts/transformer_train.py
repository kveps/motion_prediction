from models.loss.nll_loss import NLL_Loss
from models.transformer.transformer import Transformer_NN
from utils.data.motion_dataset import TransformerMotionDataset
from utils.viz.visualize_scenario import visualize_model_inputs_and_output
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import datetime

# Determine the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Create the necessary dataloaders
#
# Training
training_dataset = TransformerMotionDataset(
    "./data/uncompressed/tf_example/training/")
training_dataloader = DataLoader(training_dataset, batch_size=5, shuffle=True)
# Validation
validation_dataset = TransformerMotionDataset(
    "./data/uncompressed/tf_example/validation/")
validation_dataloader = DataLoader(
    validation_dataset, batch_size=5, shuffle=False)
# Testing
test_dataset = TransformerMotionDataset(
    "./data/uncompressed/tf_example/testing/")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Setup necessary input sizes for the model
dummy_element = training_dataset[0]
agent_input = dummy_element['agent_input']
static_roadgraph_input = dummy_element['static_roadgraph_polyline_input']
dynamic_roadgraph_input = dummy_element['dynamic_roadgraph_input']
agent_target = dummy_element['agent_target']
agent_target_valid = dummy_element['agent_target_valid']

num_agent_features = agent_input.size(dim=-1)
num_static_roadgraph_features = static_roadgraph_input.size(dim=-1)
num_dynamic_roadgraph_features = dynamic_roadgraph_input.size(dim=-1)
num_past_timesteps = agent_input.size(dim=-2)
num_future_features = agent_target.size(dim=-1)
num_future_timesteps = agent_target.size(dim=-2)
num_future_trajectories = 1
num_model_features = 256

# Setup model inputs and outputs
model = Transformer_NN(num_agent_features=num_agent_features,
                       num_static_road_features=num_static_roadgraph_features,
                       num_dynamic_road_features=num_dynamic_roadgraph_features,
                       num_past_timesteps=num_past_timesteps,
                       num_model_features=num_model_features,
                       num_future_trajectories=num_future_trajectories,
                       num_future_timesteps=num_future_timesteps,
                       num_future_features=num_future_features)
model.to(device)
print("Model has been set, num params: ", sum(p.numel()
      for p in model.parameters()))

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = NLL_Loss()

# Config
TRAINING_MODE = True
NUM_EPOCHS = 100

if TRAINING_MODE:
    # Training Loop
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()  # Set model to training mode
        train_loss = 0.0
        print("Epoch: ", epoch)
        for dataset_element in training_dataloader:
            # fetch inputs
            agents = dataset_element['agent_input'].to(device)
            agents_valid = dataset_element['agent_input_valid'].to(device)
            static_road = dataset_element['static_roadgraph_polyline_input'].to(
                device)
            static_road_valid = dataset_element['static_roadgraph_polyline_valid'].to(
                device
            )
            dynamic_road = dataset_element['dynamic_roadgraph_input'].to(
                device)
            dynamic_road_valid = dataset_element['dynamic_roadgraph_valid'].to(
                device
            )
            agent_target = dataset_element['agent_target'].to(device)
            agent_target_valid = dataset_element['agent_target_valid'].to(
                device)

            # initialize future agents and valid
            batch_size, num_agents, _, _ = agents.size()
            future_agents = torch.zeros(
                (batch_size, num_agents, num_future_timesteps, num_future_features),
                dtype=torch.float32,
                device=device,
            )
            future_agents_valid = torch.amax(
                agents_valid, dim=-1, keepdim=True).repeat(1, 1, num_future_timesteps)

            optimizer.zero_grad()
            trajectories, probs = model(agents,
                                        agents_valid,
                                        static_road,
                                        static_road_valid,
                                        dynamic_road,
                                        dynamic_road_valid,
                                        future_agents,
                                        future_agents_valid)
            loss = loss_fn(
                trajectories, probs, agent_target, agent_target_valid)
            loss.backward()
            print("Running loss: ", loss.item())
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(training_dataloader)

        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for dataset_element in validation_dataloader:
                # fetch inputs
                agents = dataset_element['agent_input'].to(device)
                agents_valid = dataset_element['agent_input_valid'].to(device)
                static_road = dataset_element['static_roadgraph_polyline_input'].to(
                    device)
                static_road_valid = dataset_element['static_roadgraph_polyline_valid'].to(
                    device
                )
                dynamic_road = dataset_element['dynamic_roadgraph_input'].to(
                    device)
                dynamic_road_valid = dataset_element['dynamic_roadgraph_valid'].to(
                    device
                )
                agent_target = dataset_element['agent_target'].to(device)
                agent_target_valid = dataset_element['agent_target_valid'].to(
                    device)

                # initialize future agents and valid
                batch_size, num_agents, _, _ = agents.size()
                future_agents = torch.zeros(
                    (batch_size, num_agents, num_future_timesteps, num_future_features),
                    dtype=torch.float32,
                    device=device,
                )
                future_agents_valid = torch.amax(
                    agents_valid, dim=-1, keepdim=True).repeat(1, 1, num_future_timesteps)

                optimizer.zero_grad()
                trajectories, probs = model(agents,
                                            agents_valid,
                                            static_road,
                                            static_road_valid,
                                            dynamic_road,
                                            dynamic_road_valid,
                                            future_agents,
                                            future_agents_valid)
                loss = loss_fn(
                    trajectories, probs, agent_target, agent_target_valid)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(validation_dataloader)

        print(
            f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        now = datetime.datetime.now()
        path = "./models/trained_weights/transformer_model_" + \
            str(epoch + 1) + "_" + now.strftime("%Y-%m-%d %H:%M:%S") + ".pt"
        torch.save(model.state_dict(), path)
else:
    # Testing
    model_path = "./models/trained_weights/transformer_model_12_2025-03-18 17:36:10.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():
        for dataset_element in test_dataloader:
            # fetch inputs
            agents = dataset_element['agent_input'].to(device)
            agents_valid = dataset_element['agent_input_valid'].to(device)
            static_road = dataset_element['static_roadgraph_polyline_input'].to(
                device)
            static_road_valid = dataset_element['static_roadgraph_polyline_valid'].to(
                device
            )
            dynamic_road = dataset_element['dynamic_roadgraph_input'].to(
                device)
            dynamic_road_valid = dataset_element['dynamic_roadgraph_valid'].to(
                device
            )
            agent_target = dataset_element['agent_target'].to(device)
            agent_target_valid = dataset_element['agent_target_valid'].to(
                device)

            # initialize future agents and valid
            batch_size, num_agents, _, _ = agents.size()
            future_agents = torch.zeros(
                (batch_size, num_agents, num_future_timesteps, num_future_features),
                dtype=torch.float32,
                device=device,
            )
            future_agents_valid = torch.amax(
                agents_valid, dim=-1, keepdim=True).repeat(1, 1, num_future_timesteps)

            optimizer.zero_grad()
            trajectories, probs = model(agents,
                                        agents_valid,
                                        static_road,
                                        static_road_valid,
                                        dynamic_road,
                                        dynamic_road_valid,
                                        future_agents,
                                        future_agents_valid)
            loss = loss_fn(
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
