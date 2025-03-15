import tensorflow as tf
import torch
from torch.utils.data import Dataset
from utils.data.features_description import get_features_description
from utils.viz.visualize_scenario import visualize_scenario_image
from utils.data.data_processing_helpers import (
    downsample_roadgraph,
    get_data_file_names,
    translate_parsed_dataset_to_av_center,
    arrange_agent_model_input,
    arrange_agent_model_target,
    arrange_dynamic_roadgraph_model_input,
    arrange_static_roadgraph_model_input,
)


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    fd = get_features_description()
    parsed = tf.io.parse_single_example(example_proto, fd)
    # Translate the data points around the AV center i.e. AV is at origin
    transformed = translate_parsed_dataset_to_av_center(parsed)
    # Downsample the roadgraph
    # TODO: Do more intelligent filtering of the data
    filtered = downsample_roadgraph(transformed)
    return filtered


class FilteredMotionDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        data_files = get_data_file_names(data_path)
        # Append the directory path to get a complete file path
        data_files = [data_path + file for file in data_files]

        # Load the tf dataset
        tf_dataset = tf.data.TFRecordDataset(data_files)
        self.parsed_tf_dataset = tf_dataset.map(_parse_function)
        self.data_files = data_files

    def __len__(self):
        # Determine the length of the TensorFlow dataset
        return len(self.data_files)

    def __getitem__(self, idx):
        # Take the element at the given index
        element = list(self.parsed_tf_dataset.skip(idx).take(1))[0]

        # Convert TensorFlow tensors to NumPy arrays
        numpy_element = {key: value.numpy() for key, value in element.items()}

        # Convert NumPy arrays to PyTorch tensors
        torch_element = {key: torch.tensor(value)
                         for key, value in numpy_element.items()}

        # Static roadgraph
        static_roadgraph_input, _ = arrange_static_roadgraph_model_input(
            torch_element)
        # Dynamic roadgraph
        dynamic_roadgraph_input, _ = arrange_dynamic_roadgraph_model_input(
            torch_element)
        # Agent states
        agent_input, _ = arrange_agent_model_input(
            torch_element)
        # Agent targets
        agent_target, agent_target_states_valid = arrange_agent_model_target(
            torch_element)

        return (agent_input,
                static_roadgraph_input,
                dynamic_roadgraph_input,
                agent_target,
                agent_target_states_valid,
                )

    def get_full_torch_element(self, idx):
        # Take the element at the given index
        element = list(self.parsed_tf_dataset.skip(idx).take(1))[0]

        # Convert TensorFlow tensors to NumPy arrays
        numpy_element = {key: value.numpy() for key, value in element.items()}

        # Convert NumPy arrays to PyTorch tensors
        torch_element = {key: torch.tensor(value)
                         for key, value in numpy_element.items()}

        return torch_element

    def get_tf_dataset(self):
        return self.parsed_tf_dataset


# directory_path = "./data/uncompressed/tf_example/training/"
# motion_dataset = FilteredMotionDataset(directory_path)
# visualize_scenario_image(motion_dataset.get_full_torch_element(10))
