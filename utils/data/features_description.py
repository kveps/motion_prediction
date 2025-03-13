import tensorflow as tf

# Consts
NUM_ROADGRAPH_SAMPLES = 30000
NUM_PAST_STATES = 10
NUM_CURRENT_STATES = 1
NUM_FUTURE_STATES = 80
NUM_AGENTS = 128
NUM_TL_STATES = 16

# Static roadgraph features
static_roadgraph_features = {
    'roadgraph_samples/dir': tf.io.FixedLenFeature(
        [NUM_ROADGRAPH_SAMPLES, 3], tf.float32, default_value=None
    ),
    'roadgraph_samples/id': tf.io.FixedLenFeature(
        [NUM_ROADGRAPH_SAMPLES, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/type': tf.io.FixedLenFeature(
        [NUM_ROADGRAPH_SAMPLES, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/valid': tf.io.FixedLenFeature(
        [NUM_ROADGRAPH_SAMPLES, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/xyz': tf.io.FixedLenFeature(
        [NUM_ROADGRAPH_SAMPLES, 3], tf.float32, default_value=None
    ),
}

# Dynamic roadgraph features
dynamic_roadgraph_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature(
            [NUM_CURRENT_STATES, NUM_TL_STATES], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature(
            [NUM_CURRENT_STATES, NUM_TL_STATES], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature(
            [NUM_CURRENT_STATES, NUM_TL_STATES], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature(
            [NUM_CURRENT_STATES, NUM_TL_STATES], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature(
            [NUM_CURRENT_STATES, NUM_TL_STATES], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature(
            [NUM_PAST_STATES, NUM_TL_STATES], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature(
            [NUM_PAST_STATES, NUM_TL_STATES], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature(
            [NUM_PAST_STATES, NUM_TL_STATES], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature(
            [NUM_PAST_STATES, NUM_TL_STATES], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature(
            [NUM_PAST_STATES, NUM_TL_STATES], tf.float32, default_value=None),
}

# Features of actors
actor_features = {
    'state/id':
        tf.io.FixedLenFeature([NUM_AGENTS], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([NUM_AGENTS], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([NUM_AGENTS], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([NUM_AGENTS], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_CURRENT_STATES], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_FUTURE_STATES],
                              tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_FUTURE_STATES],
                              tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_FUTURE_STATES],
                              tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_FUTURE_STATES], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature(
            [NUM_AGENTS, NUM_FUTURE_STATES], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_FUTURE_STATES],
                              tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_FUTURE_STATES],
                              tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_FUTURE_STATES],
                              tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_FUTURE_STATES],
                              tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_FUTURE_STATES],
                              tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_FUTURE_STATES],
                              tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_FUTURE_STATES],
                              tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([NUM_AGENTS, NUM_PAST_STATES],
                              tf.float32, default_value=None),
}


def get_features_description():
    features_description = {}
    features_description.update(static_roadgraph_features)
    features_description.update(dynamic_roadgraph_features)
    features_description.update(actor_features)

    return features_description
