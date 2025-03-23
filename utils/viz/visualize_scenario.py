from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random

import uuid

import numpy as np
import tensorflow as tf
import torch


def create_figure_and_axes(size_pixels):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax


def fig_canvas_image(fig):
    """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents):
    """Compute a color map array of shape [num_agents, 4]."""
    colors = cm.get_cmap('jet', num_agents)
    colors = colors(range(num_agents))
    np.random.shuffle(colors)
    return colors


def get_viewport(all_states, all_states_mask):
    """Gets the region containing the data.

    Args:
      all_states: states of agents as an array of shape [num_agents, num_steps,
        2].
      all_states_mask: binary mask of shape [num_agents, num_steps] for
        `all_states`.

    Returns:
      center_y: float. y coordinate for center of data.
      center_x: float. x coordinate for center of data.
      width: float. Width of data.
    """
    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width


def visualize_one_step(states,
                       mask,
                       roadgraph,
                       title,
                       center_y,
                       center_x,
                       width,
                       color_map,
                       size_pixels=1000):
    """Generate visualization for a single step."""

    # Create figure and axes.
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)

    # Plot roadgraph.
    rg_pts = roadgraph[:, :2].T
    ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)

    masked_x = states[:, 0][mask]
    masked_y = states[:, 1][mask]
    colors = color_map[mask]

    # Plot agent current position.
    ax.scatter(
        masked_x,
        masked_y,
        marker='o',
        linewidths=3,
        color=colors,
    )

    # Title.
    ax.set_title(title)

    # Set axes.  Should be at least 10m on a side and cover 160% of agents.
    size = max(10, width * 1.0)
    ax.axis([
        -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
        size / 2 + center_y
    ])
    ax.set_aspect('equal')

    image = fig_canvas_image(fig)
    plt.close(fig)
    return image


def visualize_all_agents_smooth(
    decoded_example,
    size_pixels=1000,
):
    """Visualizes all agent predicted trajectories in a serie of images.

    Args:
      decoded_example: Dictionary containing agent info about all modeled agents.
      size_pixels: The size in pixels of the output image.

    Returns:
      T of [H, W, 3] uint8 np.arrays of the drawn matplotlib's figure canvas.
    """
    # [num_agents, num_past_steps, 2] float32.
    past_states = tf.stack(
        [decoded_example['state/past/x'], decoded_example['state/past/y']],
        -1).numpy()
    past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0

    # [num_agents, 1, 2] float32.
    current_states = tf.stack(
        [decoded_example['state/current/x'], decoded_example['state/current/y']],
        -1).numpy()
    current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0

    # [num_agents, num_future_steps, 2] float32.
    future_states = tf.stack(
        [decoded_example['state/future/x'], decoded_example['state/future/y']],
        -1).numpy()
    future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0

    # [num_points, 3] float32.
    roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()

    num_agents, num_past_steps, _ = past_states.shape
    num_future_steps = future_states.shape[1]

    color_map = get_colormap(num_agents)

    # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
    all_states = np.concatenate(
        [past_states, current_states, future_states], 1)

    # [num_agens, num_past_steps + 1 + num_future_steps] float32.
    all_states_mask = np.concatenate(
        [past_states_mask, current_states_mask, future_states_mask], 1)

    center_y, center_x, width = get_viewport(all_states, all_states_mask)

    images = []

    # Generate images from past time steps.
    for i, (s, m) in enumerate(
        zip(
            np.split(past_states, num_past_steps, 1),
            np.split(past_states_mask, num_past_steps, 1))):
        im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                                'past: %d' % (num_past_steps - i), center_y,
                                center_x, width, color_map, size_pixels)
        images.append(im)

    # Generate one image for the current time step.
    s = current_states
    m = current_states_mask

    im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz, 'current', center_y,
                            center_x, width, color_map, size_pixels)
    images.append(im)

    # Generate images from future time steps.
    for i, (s, m) in enumerate(
        zip(
            np.split(future_states, num_future_steps, 1),
            np.split(future_states_mask, num_future_steps, 1))):
        im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                                'future: %d' % (
                                    i + 1), center_y, center_x, width,
                                color_map, size_pixels)
        images.append(im)

    return images


def create_animation(images):
    """ Creates a Matplotlib animation of the given images.

    Args:
      images: A list of numpy arrays representing the images.

    Returns:
      A matplotlib.animation.Animation.

    Usage:
      anim = create_animation(images)
      anim.save('/tmp/animation.avi')
      HTML(anim.to_html5_video())
    """

    plt.ioff()
    fig, ax = plt.subplots()
    dpi = 100
    size_inches = 1000 / dpi
    fig.set_size_inches([size_inches, size_inches])
    plt.ion()

    def animate_func(i):
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid('off')

    anim = animation.FuncAnimation(
        fig, animate_func, frames=len(images) // 2, interval=100)
    plt.close(fig)
    return anim


def visualize_scenario_image(scenario_tensor):
    """ Visualize full scenario as a single image

    Args:
      scenario_tensor: Torch tensors containing all details of the scenario.
    """

    ################ Agents ################

    # Past agent states
    validity_past = scenario_tensor['state/past/valid'].bool()
    valid_past_x = (scenario_tensor['state/past/x'])[validity_past]
    valid_past_y = (scenario_tensor['state/past/y'])[validity_past]
    # [num_agents, num_past_steps, 2] float32.
    past_states = torch.cat(
        (
            valid_past_x.unsqueeze(dim=-1),
            valid_past_y.unsqueeze(dim=-1),
        ),
        dim=-1
    ).detach().cpu().numpy()

    # Current agent states
    validity_current = scenario_tensor['state/current/valid'].bool()
    valid_current_x = (scenario_tensor['state/current/x'])[validity_current]
    valid_current_y = (scenario_tensor['state/current/y'])[validity_current]
    # [num_agents, num_current_steps, 2] float32.
    current_states = torch.cat(
        (
            valid_current_x.unsqueeze(dim=-1),
            valid_current_y.unsqueeze(dim=-1),
        ),
        dim=-1
    ).detach().cpu().numpy()

    # Future agent states
    validity_future = scenario_tensor['state/future/valid'].bool()
    valid_future_x = (scenario_tensor['state/future/x'])[validity_future]
    valid_future_y = (scenario_tensor['state/future/y'])[validity_future]
    # [num_agents, num_future_steps, 2] float32.
    future_states = torch.cat(
        (
            valid_future_x.unsqueeze(dim=-1),
            valid_future_y.unsqueeze(dim=-1),
        ),
        dim=-1
    ).detach().cpu().numpy()

    # Plot agent points
    plt.plot(past_states[..., 0], past_states[..., 1],
             'r.', markersize=3, label='Past Actor Points')
    plt.plot(current_states[..., 0], current_states[..., 1],
             'bo', markersize=4, label='Current Actor Point')
    plt.plot(future_states[..., 0], future_states[..., 1],
             'g.', markersize=3, label='Future Actor Points')

    ################ Static road points ################

    # Static road samples
    # [num_points, 3] float32.
    valid_road_samples = scenario_tensor['roadgraph_samples/valid'].squeeze(
        dim=1).bool()
    valid_roadgraph_xyz = (
        scenario_tensor['roadgraph_samples/xyz'])[valid_road_samples].detach().cpu().numpy()

    # Plot static road points
    plt.plot(valid_roadgraph_xyz[..., 0], valid_roadgraph_xyz[...,
             1], 'k.', markersize=0.5, label='Road Points')

    ################ Dynamic Road points ################

    # Past TL states
    validity_tl_past = scenario_tensor['traffic_light_state/past/valid'].bool()
    valid_tl_past_x = (
        scenario_tensor['traffic_light_state/past/x'])[validity_tl_past]
    valid_tl_past_y = (
        scenario_tensor['traffic_light_state/past/y'])[validity_tl_past]
    # [num_past_steps, num_tl_states] float32.
    tl_past_states = torch.cat(
        (
            valid_tl_past_x.unsqueeze(dim=-1),
            valid_tl_past_y.unsqueeze(dim=-1),
        ),
        dim=-1
    ).detach().cpu().numpy()

    # Current TL states
    validity_tl_current = scenario_tensor['traffic_light_state/current/valid'].bool()
    valid_tl_current_x = (
        scenario_tensor['traffic_light_state/current/x'])[validity_tl_current]
    valid_tl_current_y = (
        scenario_tensor['traffic_light_state/current/y'])[validity_tl_current]
    # [num_current_steps, num_tl_states] float32.
    tl_current_states = torch.cat(
        (
            valid_tl_current_x.unsqueeze(dim=-1),
            valid_tl_current_y.unsqueeze(dim=-1),
        ),
        dim=-1
    ).detach().cpu().numpy()

    # Plot Traffic light points
    plt.plot(tl_past_states[..., 0], tl_past_states[..., 1],
             'yo', markersize=7, label='Past TL states')
    plt.plot(tl_current_states[..., 0], tl_current_states[...,
             1], 'yo', markersize=7, label='Current TL states')

    # Beautify

    # Set limits
    road_x_min = min(valid_roadgraph_xyz[..., 0])
    road_x_max = max(valid_roadgraph_xyz[..., 0])
    road_y_min = min(valid_roadgraph_xyz[..., 1])
    road_y_max = max(valid_roadgraph_xyz[..., 1])
    plt.xlim(road_x_min, road_x_max)
    plt.ylim(road_y_min, road_y_max)

    # Set plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Road and Actor Visualization')
    plt.axis('equal')  # Ensure equal scaling for x and y axes
    plt.show()


def visualize_model_inputs_and_output(model_input, model_output,
                                      index_in_batch=0):
    """ Visualize model input and output as a single image

    Args:
        model_input (dict): A dictionary containing the model input.
        model_output (dict): A dictionary containing the model output.
        index_in_batch (int): The index of the batch to visualize
    """
    ############### Agents input ################

    # Setup tracks to predict
    # [num_agents].
    tracks_to_predict = model_input['tracks_to_predict']
    tracks_to_predict = (tracks_to_predict.clamp(min=0).bool())[
        index_in_batch, ...]

    # Past and current agent states
    # [num_agents, num_timesteps, 1] float32.
    agent_input_valid = model_input['agent_input_valid'].bool()
    agent_input = model_input['agent_input'][index_in_batch, ...]
    agent_input = agent_input[tracks_to_predict]
    agent_input_x = agent_input[:, :, 0]
    agent_input_y = agent_input[:, :, 1]

    # Target agent states
    # [num_agents, num_timesteps, 1] float32.
    agent_target_valid = model_input['agent_target_valid'][index_in_batch, :, :].bool(
    )
    agent_target = model_input['agent_target'][index_in_batch, ...]
    agent_target = agent_target[tracks_to_predict]
    agent_target_x = agent_target[:, :, 0]
    agent_target_y = agent_target[:, :, 1]

    # Plot agent input and target points
    plt.plot(agent_input_x, agent_input_y,
             'r.', markersize=3, label='Agent Points')
    plt.plot(agent_target_x, agent_target_y,
             'g.', markersize=3, label='Future Agent Points')

    ############### Agents output ################

    # Model ouput agent states
    # [num_agents, num_future_trajectories, num_timesteps, 3] float32.
    agent_trajs = model_output['agent_trajs'][index_in_batch, ...]
    agent_trajs = agent_trajs[tracks_to_predict]
    # [num_agents, num_future_trajectories] float32.
    agent_probs = model_output['agent_probs'][index_in_batch, ...]
    agent_probs = agent_probs[tracks_to_predict]
    # [num_agents].
    agent_highest_prob_traj = torch.argmax(agent_probs, dim=-1)

    # [num_agents, num_timesteps, 1] float32.
    agent_output_x = agent_trajs[:, 0, :, 0]
    agent_output_y = agent_trajs[:, 0, :, 1]

    print("Agent output trajs:", agent_trajs[:, :, :, :2])

    # Plot agent output points
    plt.plot(agent_output_x, agent_output_y,
             'b.', markersize=3, label='Model output points of the highest prob trajectory')

    ################ Static road points ################

    # Static road samples
    roadsamples_valid = model_input['static_roadgraph_valid'][index_in_batch, :, 0].bool(
    )
    static_roadgraph_x = model_input['static_roadgraph_input'][index_in_batch,
                                                               :, 0][roadsamples_valid]
    static_roadgraph_y = model_input['static_roadgraph_input'][index_in_batch,
                                                               :, 1][roadsamples_valid]

    # Plot static road points
    plt.plot(static_roadgraph_x, static_roadgraph_y,
             'k.', markersize=0.5, label='Road Points')

    # TODO: Add dynamic roadgraph (traffic light) points

    # Beautify

    # Set limits
    road_x_min = min(static_roadgraph_x)
    road_x_max = max(static_roadgraph_x)
    road_y_min = min(static_roadgraph_y)
    road_y_max = max(static_roadgraph_y)
    plt.xlim(road_x_min, road_x_max)
    plt.ylim(road_y_min, road_y_max)

    # Set plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Road and Actor Visualization')
    plt.axis('equal')  # Ensure equal scaling for x and y axes
    plt.show()


def visualize_polylines(polylines, validity):
    """Visualize  polylines.

    Args:
      polylines: A list of polylines, each represented as a list of
        points, where each point is a list of three floats [x, y, z] and type.

    Returns:
      A matplotlib figure.
    """
    num_polylines = polylines.shape[0]
    colors = plt.cm.get_cmap('viridis', num_polylines)

    plt.figure()

    for i in range(num_polylines):
        polyline = polylines[i]
        valid = validity[i]

        valid_x = []
        valid_y = []

        for j in range(polyline.shape[0]):
            if valid[j]:
                valid_x.append(polyline[j, 0])
                valid_y.append(polyline[j, 1])

        if valid_x:  # check if there are any valid points to plot
            random_color = (random.random(), random.random(), random.random())
            plt.plot(valid_x, valid_y, random_color)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polylines Visualization with Validity")
    plt.grid(True)
    plt.show()
