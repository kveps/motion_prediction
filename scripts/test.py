import argparse
import torch
from torch.utils.data import DataLoader

from models.loss.nll_loss import NLL_Loss
from models.lstm.lstm import LSTM_NN
from models.transformer.transformer import Transformer_NN
from utils.data.motion_dataset import LSTMMotionDataset, TransformerMotionDataset
from utils.viz.visualize_scenario import visualize_model_inputs_and_output

def main(args):
    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the necessary dataloaders
    if args.model_type == 'lstm':
        test_dataset = LSTMMotionDataset(f"./data/uncompressed/tf_example/{args.data_split}/testing/")
    elif args.model_type == 'transformer':
        test_dataset = TransformerMotionDataset(f"./data/uncompressed/tf_example/{args.data_split}/testing/")
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Setup necessary input sizes for the model
    dummy_element = test_dataset[0]
    agent_target = dummy_element['agent_target']
    num_future_features = agent_target.size(dim=-1)
    num_future_timesteps = agent_target.size(dim=-2)

    # Setup model
    if args.model_type == 'lstm':
        agent_input = dummy_element['agent_input']
        static_roadgraph_input = dummy_element['static_roadgraph_input']
        dynamic_roadgraph_input = dummy_element['dynamic_roadgraph_input']

        model = LSTM_NN(
            num_agent_features=agent_input.size(dim=-1),
            num_static_road_features=static_roadgraph_input.size(dim=-1),
            num_dynamic_road_features=dynamic_roadgraph_input.size(dim=-1),
            agent_hidden_size=args.agent_hidden_size,
            static_roadgraph_hidden_size=args.static_roadgraph_hidden_size,
            dynamic_roadgraph_hidden_size=args.dynamic_roadgraph_hidden_size,
            num_future_features=num_future_features,
            num_future_trajectories=args.num_future_trajectories,
            num_future_timesteps=num_future_timesteps
        )
    elif args.model_type == 'transformer':
        agent_input = dummy_element['agent_input']
        static_roadgraph_input = dummy_element['static_roadgraph_polyline_input']

        model = Transformer_NN(
            num_agent_features=agent_input.size(dim=-1),
            num_static_road_features=static_roadgraph_input.size(dim=-1),
            num_dynamic_road_features=dummy_element['dynamic_roadgraph_input'].size(dim=-1),
            num_past_timesteps=agent_input.size(dim=-2),
            num_model_features=args.num_model_features,
            num_future_trajectories=args.num_future_trajectories,
            num_future_timesteps=num_future_timesteps,
            num_future_features=num_future_features
        )

    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    print("Model has been loaded, num params: ", sum(p.numel() for p in model.parameters()))

    # Loss
    loss_fn = NLL_Loss()

    # Testing
    test_loss = 0.0
    with torch.no_grad():
        for dataset_element in test_dataloader:
            if args.model_type == 'lstm':
                agent_input = dataset_element['agent_input'].to(device)
                static_roadgraph_input = dataset_element['static_roadgraph_input'].to(device)
                dynamic_roadgraph_input = dataset_element['dynamic_roadgraph_input'].to(device)
                agent_target = dataset_element['agent_target'].to(device)
                agent_target_valid = dataset_element['agent_target_valid'].to(device)

                trajectories, probs = model(agent_input, static_roadgraph_input, dynamic_roadgraph_input)

            elif args.model_type == 'transformer':
                agents = dataset_element['agent_input'].to(device)
                agents_valid = dataset_element['agent_input_valid'].to(device)
                static_road = dataset_element['static_roadgraph_polyline_input'].to(device)
                static_road_valid = dataset_element['static_roadgraph_polyline_valid'].to(device)
                dynamic_road = dataset_element['dynamic_roadgraph_input'].to(device)
                dynamic_road_valid = dataset_element['dynamic_roadgraph_valid'].to(device)
                agent_target = dataset_element['agent_target'].to(device)
                agent_target_valid = dataset_element['agent_target_valid'].to(device)

                batch_size, num_agents, _, _ = agents.size()
                future_agents = torch.zeros((batch_size, num_agents, num_future_timesteps, num_future_features), dtype=torch.float32, device=device)
                future_agents_valid = torch.amax(agents_valid, dim=-1, keepdim=True).repeat(1, 1, num_future_timesteps)

                trajectories, probs = model(agents, agents_valid, static_road, static_road_valid, dynamic_road, dynamic_road_valid, future_agents, future_agents_valid)

            loss = loss_fn(trajectories, probs, agent_target, agent_target_valid)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'transformer'], help='Type of model to test')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--data_split', type=str, default='all', help='Data split to use for testing')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing (1 for visualization)')
    parser.add_argument('--num_future_trajectories', type=int, default=1, help='Number of future trajectories to predict')

    # LSTM specific args (needed for model instantiation)
    parser.add_argument('--agent_hidden_size', type=int, default=32, help='LSTM agent hidden size')
    parser.add_argument('--static_roadgraph_hidden_size', type=int, default=64, help='LSTM static roadgraph hidden size')
    parser.add_argument('--dynamic_roadgraph_hidden_size', type=int, default=32, help='LSTM dynamic roadgraph hidden size')

    # Transformer specific args (needed for model instantiation)
    parser.add_argument('--num_model_features', type=int, default=256, help='Transformer model feature dimension')

    args = parser.parse_args()
    main(args)
