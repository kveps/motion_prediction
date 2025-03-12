import os
import tensorflow as tf

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
        file_names = [entry for entry in all_entries if os.path.isfile(os.path.join(dir_path, entry))]
        return file_names
    except FileNotFoundError:
        return f"Directory not found: {dir_path}"
    except NotADirectoryError:
         return f"Not a directory: {dir_path}"

def translate_parsed_dataset_to_av_center(parsed):
    """
        Translates all relevant points such that AV is at the origin (0,0)
    """
    transformed = {}  # Create a new dictionary

    # Get AV index in the set of agents
    av_idx = 0
    for i in range(parsed['state/current/x'].shape[0]):
        if parsed['state/is_sdc'][i] == 1:
            av_idx = i
    assert(av_idx != None)

    # Set AV center
    av_center_x = parsed['state/current/x'][av_idx, parsed['state/current/x'].shape[1] - 1]
    av_center_y = parsed['state/current/y'][av_idx, parsed['state/current/x'].shape[1] - 1]   
    av_center_z = 0.0
    av_center_xyz = tf.stack([av_center_x, av_center_y, av_center_z])

    for key, value in parsed.items():
        if key == 'state/past/x':
            transformed[key] = tf.subtract(parsed['state/past/x'], av_center_x)
        elif key == 'state/past/y':
            transformed[key] = tf.subtract(parsed['state/past/y'], av_center_y)
        elif key == 'state/current/x':
            transformed[key] = tf.subtract(parsed['state/current/x'], av_center_x)
        elif key == 'state/current/y':
            transformed[key] = tf.subtract(parsed['state/current/y'], av_center_y)
        elif key == 'state/future/x':
            transformed[key] = tf.subtract(parsed['state/future/x'], av_center_x)
        elif key == 'state/future/y':
            transformed[key] = tf.subtract(parsed['state/future/y'], av_center_y)
        elif key == 'roadgraph_samples/xyz':
            transformed['roadgraph_samples/xyz'] = tf.subtract(parsed['roadgraph_samples/xyz'], av_center_xyz)
        elif key == 'traffic_light_state/current/x':
            transformed[key] = tf.subtract(parsed['traffic_light_state/current/x'], av_center_x)
        elif key == 'traffic_light_state/current/y':
            transformed[key] = tf.subtract(parsed['traffic_light_state/current/y'], av_center_y)
        elif key == 'traffic_light_state/current/z':
            transformed[key] = tf.subtract(parsed['traffic_light_state/current/x'], av_center_z)
        elif key == 'traffic_light_state/past/x':
            transformed[key] = tf.subtract(parsed['traffic_light_state/past/x'], av_center_x)
        elif key == 'traffic_light_state/past/y':
            transformed[key] = tf.subtract(parsed['traffic_light_state/past/y'], av_center_y)
        elif key == 'traffic_light_state/past/z':
            transformed[key] = tf.subtract(parsed['traffic_light_state/past/x'], av_center_z)
        else:
            transformed[key] = value  # Copy unchanged tensors

    return transformed