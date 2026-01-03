"""
Transformer training script with support for both local and Google Colab (GCS) training.

Usage:
    # Local training (default)
    python transformer_train.py
    
    # Colab training with GCS paths
    python transformer_train.py --colab
    
    # Testing mode
    python transformer_train.py --test --model-path <path_to_model>
"""
from models.loss.nll_loss import NLL_Loss
from models.transformer.transformer import Transformer_NN
from utils.data.motion_dataset import MotionDataset
from utils.viz.visualize_scenario import visualize_model_inputs_and_output
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import datetime
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train or test Transformer model')
parser.add_argument('--colab', action='store_true', 
                    help='Use Google Colab mode with GCS paths (requires authentication)')
parser.add_argument('--test', action='store_true',
                    help='Run in testing mode instead of training')
parser.add_argument('--model-path', type=str, default=None,
                    help='Path to model weights for testing')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs (default: 100)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size for training/validation (default: 16)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate (default: 0.01)')
args = parser.parse_args()

# Determine the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure paths based on mode
if args.colab:
    print("Running in Colab mode - using GCS paths")
    TRAINING_PATH = "gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/training/"
    VALIDATION_PATH = "gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/validation/"
    TESTING_PATH = "gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/testing/"
    
    # Try Google Drive first, fallback to /content if not mounted
    if os.path.exists("/content/drive/MyDrive"):
        SAVE_DIR = "/content/drive/MyDrive/av_prediction/models/trained_weights/"
        print("✓ Using Google Drive for model storage")
    else:
        SAVE_DIR = "/content/models/trained_weights/"
        print("⚠ Google Drive not mounted - saving to /content/ (temporary storage)")
else:
    print("Running in local mode")
    TRAINING_PATH = "./data/uncompressed/tf_example/training/"
    VALIDATION_PATH = "./data/uncompressed/tf_example/validation/"
    TESTING_PATH = "./data/uncompressed/tf_example/testing/"
    SAVE_DIR = "./models/trained_weights/"

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Models will be saved to: {SAVE_DIR}")

# Create the necessary dataloaders
print("Loading datasets...")
training_dataset = MotionDataset(TRAINING_PATH)
training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size)

validation_dataset = MotionDataset(VALIDATION_PATH)
validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size)

test_dataset = MotionDataset(TESTING_PATH)
test_dataloader = DataLoader(test_dataset, batch_size=1)

# Setup necessary input sizes for the model
dummy_element = training_dataset[0]
agent_input_continuous = dummy_element['agent_input_continuous']
static_roadgraph_input = dummy_element['static_roadgraph_polyline_input']
dynamic_roadgraph_continuous = dummy_element['dynamic_roadgraph_continuous']
agent_target = dummy_element['agent_target']

# Get feature dimensions (continuous only, categorical are embedded separately)
num_agent_continuous_features = agent_input_continuous.size(dim=-1)
num_static_roadgraph_features = static_roadgraph_input.size(dim=-1)
num_dynamic_roadgraph_continuous_features = dynamic_roadgraph_continuous.size(dim=-1)
num_past_timesteps = agent_input_continuous.size(dim=-2)
num_future_features = agent_target.size(dim=-1)
num_future_timesteps = agent_target.size(dim=-2)
num_future_trajectories = 1
num_model_features = 256
categorical_embedding_dim = 16

# Setup model inputs and outputs
model = Transformer_NN(num_agent_features=num_agent_continuous_features,
                       num_static_road_features=num_static_roadgraph_features,
                       num_dynamic_road_features=num_dynamic_roadgraph_continuous_features,
                       num_past_timesteps=num_past_timesteps,
                       num_model_features=num_model_features,
                       categorical_embedding_dim=categorical_embedding_dim,
                       num_future_trajectories=num_future_trajectories,
                       num_future_timesteps=num_future_timesteps,
                       num_future_features=num_future_features)
model.to(device)
print("Model has been set, num params: ", sum(p.numel()
      for p in model.parameters()))

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_fn = NLL_Loss()

# Config
NUM_EPOCHS = args.epochs

if not args.test:
    # Training Loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()  # Set model to training mode
        train_loss = 0.0
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        for batch_idx, dataset_element in enumerate(training_dataloader):
            # Fetch separated continuous and categorical features
            agents_cont = dataset_element['agent_input_continuous'].to(device)
            agents_cat = dataset_element['agent_input_categorical'].to(device)
            agents_valid = dataset_element['agent_input_valid'].to(device)
            static_road = dataset_element['static_roadgraph_polyline_input'].to(device)
            static_road_valid = dataset_element['static_roadgraph_polyline_valid'].to(device)
            dyn_road_cont = dataset_element['dynamic_roadgraph_continuous'].to(device)
            dyn_road_cat = dataset_element['dynamic_roadgraph_categorical'].to(device)
            dyn_road_valid = dataset_element['dynamic_roadgraph_valid'].to(device)
            agent_target = dataset_element['agent_target'].to(device)
            agent_target_valid = dataset_element['agent_target_valid'].to(device)
            tracks_to_predict = dataset_element['tracks_to_predict'].to(device)

            # Initialize future agents
            batch_size, num_agents, _, _ = agents_cont.size()
            future_agents = torch.randn(
                (batch_size, num_agents, num_future_trajectories, num_model_features),
                dtype=torch.float32, device=device)
            future_agents_valid = torch.ones([batch_size, num_agents, num_future_trajectories], dtype=torch.float32, device=device)                

            optimizer.zero_grad()
            trajectories, probs = model(
                agents_cont, agents_cat, agents_valid,
                static_road, static_road_valid,
                dyn_road_cont, dyn_road_cat, dyn_road_valid,
                future_agents, future_agents_valid
            )
            loss = loss_fn(trajectories, probs, agent_target, agent_target_valid, tracks_to_predict)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}, \
                Done with {batch_idx*args.batch_size+args.batch_size} samples\
                Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(training_dataloader)

        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for dataset_element in validation_dataloader:
                # Fetch separated continuous and categorical features
                agents_cont = dataset_element['agent_input_continuous'].to(device)
                agents_cat = dataset_element['agent_input_categorical'].to(device)
                agents_valid = dataset_element['agent_input_valid'].to(device)
                static_road = dataset_element['static_roadgraph_polyline_input'].to(device)
                static_road_valid = dataset_element['static_roadgraph_polyline_valid'].to(device)
                dyn_road_cont = dataset_element['dynamic_roadgraph_continuous'].to(device)
                dyn_road_cat = dataset_element['dynamic_roadgraph_categorical'].to(device)
                dyn_road_valid = dataset_element['dynamic_roadgraph_valid'].to(device)
                agent_target = dataset_element['agent_target'].to(device)
                agent_target_valid = dataset_element['agent_target_valid'].to(device)
                tracks_to_predict = dataset_element['tracks_to_predict'].to(device)

                # Initialize future agents
                batch_size, num_agents, _, _ = agents_cont.size()
                future_agents = torch.randn(
                    (batch_size, num_agents, num_future_trajectories, num_model_features),
                    dtype=torch.float32, device=device)
                future_agents_valid = torch.ones([batch_size, num_agents, num_future_trajectories], dtype=torch.float32, device=device)

                trajectories, probs = model(
                    agents_cont, agents_cat, agents_valid,
                    static_road, static_road_valid,
                    dyn_road_cont, dyn_road_cat, dyn_road_valid,
                    future_agents, future_agents_valid
                )
                loss = loss_fn(trajectories, probs, agent_target, agent_target_valid, tracks_to_predict)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(validation_dataloader)

        print(f'\n{"="*50}')
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss:   {avg_val_loss:.4f}')
        print(f'{"="*50}')

        # Save model
        now = datetime.datetime.now()
        filename = f"transformer_model_epoch_{epoch+1}_{now.strftime('%Y%m%d_%H%M%S')}.pt"
        path = os.path.join(SAVE_DIR, filename)
        torch.save(model.state_dict(), path)
        print(f"Model saved to: {path}")
    
    print("\n✓ Training complete!")
else:
    # Testing
    if args.model_path is None:
        raise ValueError("Must specify --model-path for testing mode")
    
    print(f"\nLoading model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():
        for dataset_element in test_dataloader:
            # Fetch separated continuous and categorical features
            agents_cont = dataset_element['agent_input_continuous'].to(device)
            agents_cat = dataset_element['agent_input_categorical'].to(device)
            agents_valid = dataset_element['agent_input_valid'].to(device)
            static_road = dataset_element['static_roadgraph_polyline_input'].to(device)
            static_road_valid = dataset_element['static_roadgraph_polyline_valid'].to(device)
            dyn_road_cont = dataset_element['dynamic_roadgraph_continuous'].to(device)
            dyn_road_cat = dataset_element['dynamic_roadgraph_categorical'].to(device)
            dyn_road_valid = dataset_element['dynamic_roadgraph_valid'].to(device)
            agent_target = dataset_element['agent_target'].to(device)
            agent_target_valid = dataset_element['agent_target_valid'].to(device)
            tracks_to_predict = dataset_element['tracks_to_predict'].to(device)

            # Initialize future agents
            batch_size, num_agents, _, _ = agents_cont.size()
            future_agents = torch.randn(
                (batch_size, num_agents,  num_future_trajectories, num_model_features),
                dtype=torch.float32, device=device)
            future_agents_valid = torch.ones([batch_size, num_agents, num_future_trajectories], dtype=torch.float32, device=device)

            trajectories, probs = model(
                agents_cont, agents_cat, agents_valid,
                static_road, static_road_valid,
                dyn_road_cont, dyn_road_cat, dyn_road_valid,
                future_agents, future_agents_valid
            )
            loss = loss_fn(trajectories, probs, agent_target, agent_target_valid, tracks_to_predict)
            test_loss += loss.item()

            # Visualize the model inputs and outputs
            model_output = {
                'agent_trajs': trajectories,
                'agent_probs': probs,
            }
            visualize_model_inputs_and_output(dataset_element, model_output)

    avg_test_loss = test_loss / len(test_dataloader)

    print(f'\nTest Loss: {avg_test_loss:.4f}')
