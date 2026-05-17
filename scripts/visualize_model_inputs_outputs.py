"""
Script to visualize Transformer model inputs and outputs.

This script loads data from the Waymo motion dataset and visualizes both the
model inputs and outputs for better understanding of the prediction task.

Usage:
    # Local visualization (default)
    python visualize_model_inputs_outputs.py
    
    # Colab visualization with GCS paths
    python visualize_model_inputs_outputs.py --colab
    
    # With custom model path
    python visualize_model_inputs_outputs.py --model-path <path_to_model>
    
    # Visualize specific number of samples
    python visualize_model_inputs_outputs.py --num-samples 5
"""
from models.transformer.transformer import Transformer_NN
from models.loss.nll_loss import NLL_Loss
from utils.data.motion_dataset import MotionDataset
from utils.model.engineered_predictions import ballistic_trajectories
from utils.viz.visualize_scenario import visualize_model_inputs_and_output
from torch.utils.data import DataLoader
import torch
import argparse
import os
import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize Transformer model inputs and outputs')
parser.add_argument('--colab', action='store_true',
                    help='Use Google Colab mode with GCS paths (requires authentication)')
parser.add_argument('--local-data', action='store_true',
                    help='Read TFRecords from ./local_data/ instead of the gcsfuse mount at ./data/')
parser.add_argument('--model-path', type=str, default=None,
                    help='Path to model weights for visualization (optional)')
parser.add_argument('--num-samples', type=int, default=1,
                    help='Number of samples to visualize (default: 5)')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Batch size for visualization (default: 1)')
parser.add_argument('--data-split', type=str, default='validation',
                    choices=['training', 'validation', 'testing'],
                    help='Which data split to visualize (default: training)')
parser.add_argument('--show-ballistic', action='store_true',
                    help='Show constant-velocity ballistic predictions instead of model output')
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
elif args.local_data:
    print("Running in local-data mode (./local_data/)")
    TRAINING_PATH = "./local_data/training/"
    VALIDATION_PATH = "./local_data/validation/"
    TESTING_PATH = "./local_data/testing/"
else:
    print("Running in local mode (gcsfuse mount)")
    TRAINING_PATH = "./data/uncompressed/tf_example/training/"
    VALIDATION_PATH = "./data/uncompressed/tf_example/validation/"
    TESTING_PATH = "./data/uncompressed/tf_example/testing/"

# Select the appropriate data path
if args.data_split == 'training':
    data_path = TRAINING_PATH
elif args.data_split == 'validation':
    data_path = VALIDATION_PATH
else:
    data_path = TESTING_PATH

print(f"Loading {args.data_split} dataset from: {data_path}")

# Create the dataset and dataloader
dataset = MotionDataset(data_path)
dataloader = DataLoader(dataset, batch_size=args.batch_size)

# Setup necessary input sizes for the model
print("\nExtracting model configuration from dataset...")
dummy_element = dataset[0]
agent_input_continuous = dummy_element['agent_input_continuous']
static_roadgraph_input = dummy_element['static_roadgraph_polyline_input']
dynamic_roadgraph_continuous = dummy_element['dynamic_roadgraph_continuous']
agent_target = dummy_element['agent_target']

# Get feature dimensions
num_agent_continuous_features = agent_input_continuous.size(dim=-1)
num_static_roadgraph_features = static_roadgraph_input.size(dim=-1)
num_dynamic_roadgraph_continuous_features = dynamic_roadgraph_continuous.size(dim=-1)
num_past_timesteps = agent_input_continuous.size(dim=-2)
num_future_features = agent_target.size(dim=-1)
num_future_timesteps = agent_target.size(dim=-2)
num_future_trajectories = 1
num_model_features = 256
categorical_embedding_dim = 16

print(f"  Agent continuous features: {num_agent_continuous_features}")
print(f"  Static roadgraph features: {num_static_roadgraph_features}")
print(f"  Dynamic roadgraph features: {num_dynamic_roadgraph_continuous_features}")
print(f"  Past timesteps: {num_past_timesteps}")
print(f"  Future timesteps: {num_future_timesteps}")

# Create model
print("\nInitializing Transformer model...")
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
print(f"Model has been initialized with {sum(p.numel() for p in model.parameters())} parameters")

# Load model weights if provided
if args.model_path:
    if os.path.exists(args.model_path):
        print(f"\nLoading model weights from: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("✓ Model weights loaded successfully")
    else:
        print(f"⚠ Warning: Model path not found: {args.model_path}")
        print("  Proceeding with untrained model")
else:
    print("\n⚠ No model weights provided - using untrained model")
    print("  Tip: Use --model-path to load pre-trained weights")

# Set model to evaluation mode
model.eval()
loss_fn = NLL_Loss()

# Create tmp directory for saving plots
tmp_dir = "./tmp"
os.makedirs(tmp_dir, exist_ok=True)
print(f"Plots will be saved to: {tmp_dir}/\n")

# Visualize samples
print(f"{'='*60}")
print("Starting visualization of model inputs and outputs")
print(f"{'='*60}\n")

with torch.no_grad():
    for batch_idx, dataset_element in enumerate(dataloader):
        if batch_idx >= args.num_samples:
            break
        
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
        is_sdc = dataset_element['is_sdc'].to(device)
        tracks_to_predict = dataset_element['tracks_to_predict'].to(device)

        if args.show_ballistic:
            # [B, A, T, 4] → [B, A, 1, T, 4] (single mode)
            trajectories = ballistic_trajectories(
                agents_cont, num_future_timesteps).unsqueeze(2)
            probs = torch.ones(*trajectories.shape[:3], device=device)
        else:
            # Forward pass through model
            trajectories, probs = model(
                agents_cont, agents_cat, agents_valid,
                static_road, static_road_valid,
                dyn_road_cont, dyn_road_cat, dyn_road_valid,
            )

        # Prepare model output dictionary
        model_output = {
            'agent_trajs': trajectories,
            'agent_probs': probs,
        }

        # Prepare model input dictionary in the format expected by visualization function
        model_input = {
            'agent_input': agents_cont,
            'agent_input_valid': agents_valid,
            'agent_target': agent_target,
            'agent_target_valid': agent_target_valid,
            'static_roadgraph_input': static_road,
            'static_roadgraph_valid': static_road_valid,
            'dynamic_roadgraph_continuous': dyn_road_cont,
            'dynamic_roadgraph_valid': dyn_road_valid,
            'is_sdc': is_sdc,
            'tracks_to_predict': tracks_to_predict,
        }

        # Compute and print loss
        loss = loss_fn(trajectories, probs, agent_target, agent_target_valid, tracks_to_predict)
        print(f"Sample {batch_idx + 1} loss: {loss.item():.4f}")

        # Create save path with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(tmp_dir, f"visualization_sample_{batch_idx + 1}_{timestamp}.png")
        visualize_model_inputs_and_output(model_input, model_output, index_in_batch=0, should_visualize_outputs=True, save_path=save_path)

print(f"\n{'='*60}")
print("✓ Visualization complete!")
print(f"{'='*60}")
