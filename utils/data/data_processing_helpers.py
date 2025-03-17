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


def downsample_roadgraph(tf_dataset):
    """
    Downsample the roadgraph of a given tf dataset.

    Selects every NUM_POINTS_TO_FILTER'th point in the roadgraph, and returns
    a new dictionary with these points.

    Args:
        tf_dataset (dict): A tf dataset element, containing
        the roadgraph and other data.

    Returns:
        dict: A new dictionary with the downsampled roadgraph.
    """
    NUM_POINTS_TO_FILTER = 5
    filtered = {}  # Create a new dictionary

    for key, value in tf_dataset.items():
        # Downsample the roadgraph
        if 'roadgraph_samples/' in key:
            # Create a boolean mask to select every nth row
            mask = tf.range(value.shape[0]) % NUM_POINTS_TO_FILTER == 0
            # Apply the mask to the tensor
            filtered[key] = value[mask]
        else:
            filtered[key] = value  # Copy unchanged tensors

    return filtered


def arrange_static_roadgraph_model_input(torch_dataset_element):
    """
    Arrange the model input for the static roadgraph model.

    Args:
        torch_dataset_element (dict): A PyTorch dataset element, containing
        the roadgraph and other data.

    Returns:
        tuple: A tuple containing the static roadgraph model input and the
        valid flag for each map sample.
    """
    # [num_map_samples, 1]
    roadgraph_samples_valid = torch_dataset_element['roadgraph_samples/valid']
    roadgraph_samples_x = torch_dataset_element['roadgraph_samples/xyz'][:, 0].unsqueeze(
        dim=-1) * roadgraph_samples_valid
    roadgraph_samples_y = torch_dataset_element['roadgraph_samples/xyz'][:, 1].unsqueeze(
        dim=-1) * roadgraph_samples_valid
    roadgraph_samples_dir_z = torch_dataset_element['roadgraph_samples/dir'][:, 2].unsqueeze(
        dim=-1) * roadgraph_samples_valid
    roadgraph_samples_type = torch_dataset_element['roadgraph_samples/type'] * \
        roadgraph_samples_valid
    # [num_map_samples, 4]
    static_roadgraph_input = torch.cat(
        (roadgraph_samples_x, roadgraph_samples_y,
            roadgraph_samples_dir_z, roadgraph_samples_type), dim=-1
    )

    return static_roadgraph_input, roadgraph_samples_valid


def arrange_dynamic_roadgraph_model_input(torch_dataset_element):
    """
    Arrange the model input for the dynamic roadgraph model.

    Args:
        torch_dataset_element (dict): A PyTorch dataset element, containing
        the roadgraph and other data.

    Returns:
        tuple: A tuple containing the dynamic roadgraph model input and the
        valid flag for each map sample.
    """
    # [num_tl_states, num_past_states + num_current_states]
    traffic_light_input_states_valid = torch.cat(
        (torch_dataset_element['traffic_light_state/past/valid'].swapaxes(0, 1),
            torch_dataset_element['traffic_light_state/current/valid'].swapaxes(0, 1)), dim=-1
    )
    traffic_light_input_states_x = torch.cat(
        (torch_dataset_element['traffic_light_state/past/x'].swapaxes(0, 1),
            torch_dataset_element['traffic_light_state/current/x'].swapaxes(0, 1)), dim=-1
    ) * traffic_light_input_states_valid
    traffic_light_input_states_y = torch.cat(
        (torch_dataset_element['traffic_light_state/past/y'].swapaxes(0, 1),
            torch_dataset_element['traffic_light_state/current/y'].swapaxes(0, 1)), dim=-1
    ) * traffic_light_input_states_valid
    traffic_light_input_states_state = torch.cat(
        (torch_dataset_element['traffic_light_state/past/state'].swapaxes(0, 1),
            torch_dataset_element['traffic_light_state/current/state'].swapaxes(0, 1)), dim=-1
    ) * traffic_light_input_states_valid
    # [num_tl_states, num_past_states + num_current_states, 3]
    dynamic_roadgraph_input = torch.cat(
        (traffic_light_input_states_x.unsqueeze(dim=-1),
         traffic_light_input_states_y.unsqueeze(dim=-1),
         traffic_light_input_states_state.unsqueeze(dim=-1)), dim=-1
    )

    return dynamic_roadgraph_input, traffic_light_input_states_valid


def arrange_agent_model_input(torch_dataset_element):
    """
    Arrange the model input for the agent model.

    Args:
        torch_dataset_element (dict): A PyTorch dataset element, containing
        the agent and other data.

    Returns:
        tuple: A tuple containing the agent model input and the
        valid flag for each agent.
    """
    # [num_agents, num_past_states + num_current_states, 1]
    agent_input_states_valid = torch.cat(
        (torch_dataset_element['state/past/valid'],
         torch_dataset_element['state/current/valid']), dim=-1
    )
    agent_input_states_x = torch.cat(
        (torch_dataset_element['state/past/x'],
         torch_dataset_element['state/current/x']), dim=-1
    ) * agent_input_states_valid
    agent_input_states_y = torch.cat(
        (torch_dataset_element['state/past/y'],
         torch_dataset_element['state/current/y']), dim=-1
    ) * agent_input_states_valid
    agent_input_states_bbox_yaw = torch.cat(
        (torch_dataset_element['state/past/bbox_yaw'],
         torch_dataset_element['state/current/bbox_yaw']), dim=-1
    ) * agent_input_states_valid
    agent_input_states_velocity_x = torch.cat(
        (torch_dataset_element['state/past/velocity_x'],
         torch_dataset_element['state/current/velocity_x']), dim=-1
    ) * agent_input_states_valid
    agent_input_states_velocity_y = torch.cat(
        (torch_dataset_element['state/past/velocity_y'],
         torch_dataset_element['state/current/velocity_y']), dim=-1
    ) * agent_input_states_valid
    agent_input_states_vel_yaw = torch.cat(
        (torch_dataset_element['state/past/vel_yaw'],
         torch_dataset_element['state/current/vel_yaw']), dim=-1
    ) * agent_input_states_valid
    agent_input_states_length = torch.cat(
        (torch_dataset_element['state/past/length'],
         torch_dataset_element['state/current/length']), dim=-1
    ) * agent_input_states_valid
    agent_input_states_width = torch.cat(
        (torch_dataset_element['state/past/width'],
         torch_dataset_element['state/current/width']), dim=-1
    ) * agent_input_states_valid
    agent_input_states_type = (torch_dataset_element['state/type'].unsqueeze(dim=-1)).expand_as(
        agent_input_states_valid
    ) * agent_input_states_valid

    # [num_agents, num_past_states + num_current_states, 8]
    agent_input = torch.cat(
        (agent_input_states_x.unsqueeze(dim=-1),
         agent_input_states_y.unsqueeze(dim=-1),
         agent_input_states_bbox_yaw.unsqueeze(dim=-1),
         agent_input_states_velocity_x.unsqueeze(
            dim=-1), agent_input_states_velocity_y.unsqueeze(dim=-1),
         agent_input_states_vel_yaw.unsqueeze(dim=-1),
         agent_input_states_length.unsqueeze(
            dim=-1), agent_input_states_width.unsqueeze(dim=-1),
         agent_input_states_type.unsqueeze(dim=-1)), dim=-1
    )

    return agent_input, agent_input_states_valid


def arrange_agent_model_target(torch_dataset_element):
    """
    Arrange the model target for the agent model.

    Args:
        torch_dataset_element (dict): A PyTorch dataset element, containing
        the agent and other data.

    Returns:
        tuple: A tuple containing the agent model target and the
        valid flag for each agent.
    """
    # [num_agents, num_future_states, 1]
    agent_target_states_valid = torch_dataset_element['state/future/valid']
    agent_target_states_x = torch_dataset_element['state/future/x'] * \
        agent_target_states_valid
    agent_target_states_y = torch_dataset_element['state/future/y'] * \
        agent_target_states_valid
    agent_target_states_bbox_yaw = torch_dataset_element['state/future/bbox_yaw'] * \
        agent_target_states_valid
    # [num_agents, num_future_states, 3]
    agent_target = torch.cat(
        (agent_target_states_x.unsqueeze(dim=-1), agent_target_states_y.unsqueeze(dim=-1),
         agent_target_states_bbox_yaw.unsqueeze(dim=-1)), dim=-1
    )

    return agent_target, agent_target_states_valid
