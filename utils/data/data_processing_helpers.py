import os
import tensorflow as tf
import torch


def get_data_file_names(dir_path):
    """
    Returns a list of all file names in the specified directory.

    Args:
        dir_path (str): The path to the directory.

    Returns:
        list: A list of file names.
    """
    try:
        all_entries = os.listdir(dir_path)
        file_names = [entry for entry in all_entries if
                      os.path.isfile(os.path.join(dir_path, entry))]
        return file_names
    except FileNotFoundError:
        return f"Directory not found: {dir_path}"
    except NotADirectoryError:
        return f"Not a directory: {dir_path}"


def translate_parsed_dataset_to_av_center(tf_dataset):
    """
    Translates all spatial features in the dataset such that the autonomous vehicle (AV) 
    is centered at the origin.

    This function adjusts the positions of agents, roadgraph, and traffic light states 
    relative to the AV's last known current position. The resulting transformation ensures 
    that the AV is at the origin (0, 0, 0) in the coordinate system.

    Args:
        tf_dataset (dict): A dictionary containing TensorFlow tensors of various 
                           features including agent states, roadgraph, and traffic 
                           light states.

    Returns:
        dict: A new dictionary with the same keys as `tf_dataset`, but with 
              translated spatial features.
    """

    transformed = {}  # Create a new dictionary

    # Get AV index in the set of agents
    av_idx = 0
    for i in range(tf_dataset['state/current/x'].shape[0]):
        if tf_dataset['state/is_sdc'][i] == 1:
            av_idx = i
    assert (av_idx is not None)

    # Set AV center
    num_current_states = tf_dataset['state/current/x'].shape[1]
    av_center_x = tf_dataset['state/current/x'][av_idx,
                                                num_current_states - 1]
    av_center_y = tf_dataset['state/current/y'][av_idx,
                                                num_current_states - 1]
    av_center_z = 0.0
    av_center_xyz = tf.stack([av_center_x, av_center_y, av_center_z])

    for key, value in tf_dataset.items():
        if key == 'state/past/x':
            transformed[key] = tf.subtract(
                tf_dataset['state/past/x'], av_center_x)
        elif key == 'state/past/y':
            transformed[key] = tf.subtract(
                tf_dataset['state/past/y'], av_center_y)
        elif key == 'state/current/x':
            transformed[key] = tf.subtract(
                tf_dataset['state/current/x'], av_center_x)
        elif key == 'state/current/y':
            transformed[key] = tf.subtract(
                tf_dataset['state/current/y'], av_center_y)
        elif key == 'state/future/x':
            transformed[key] = tf.subtract(
                tf_dataset['state/future/x'], av_center_x)
        elif key == 'state/future/y':
            transformed[key] = tf.subtract(
                tf_dataset['state/future/y'], av_center_y)
        elif key == 'roadgraph_samples/xyz':
            transformed['roadgraph_samples/xyz'] = tf.subtract(
                tf_dataset['roadgraph_samples/xyz'], av_center_xyz)
        elif key == 'traffic_light_state/current/x':
            transformed[key] = tf.subtract(
                tf_dataset['traffic_light_state/current/x'], av_center_x)
        elif key == 'traffic_light_state/current/y':
            transformed[key] = tf.subtract(
                tf_dataset['traffic_light_state/current/y'], av_center_y)
        elif key == 'traffic_light_state/current/z':
            transformed[key] = tf.subtract(
                tf_dataset['traffic_light_state/current/x'], av_center_z)
        elif key == 'traffic_light_state/past/x':
            transformed[key] = tf.subtract(
                tf_dataset['traffic_light_state/past/x'], av_center_x)
        elif key == 'traffic_light_state/past/y':
            transformed[key] = tf.subtract(
                tf_dataset['traffic_light_state/past/y'], av_center_y)
        elif key == 'traffic_light_state/past/z':
            transformed[key] = tf.subtract(
                tf_dataset['traffic_light_state/past/x'], av_center_z)
        else:
            transformed[key] = value  # Copy unchanged tensors

    return transformed


def downsample_roadgraph(torch_dataset_element):
    """
    Downsample the roadgraph of a given dataset element.

    Selects every NUM_POINTS_TO_FILTER'th point in the roadgraph, and returns
    a new dictionary with these points.

    Args:
        torch_dataset_element (dict): A PyTorch dataset element, containing
        the roadgraph and other data.

    Returns:
        dict: A new dictionary with the downsampled roadgraph.
    """
    NUM_POINTS_TO_FILTER = 5
    filtered = {}  # Create a new dictionary

    for key, value in torch_dataset_element.items():
        # Downsample the roadgraph
        if 'roadgraph_samples/' in key:
            # Create a boolean mask to select every 10th row
            mask = torch.arange(value.shape[0]) % NUM_POINTS_TO_FILTER == 0
            # Apply the mask to the tensor
            filtered[key] = value[mask]
        else:
            filtered[key] = value  # Copy unchanged tensors

    return filtered
