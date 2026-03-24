import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



TENSOR_FOLDER = "/home/anders/ros2_ws/src/ur5e_whip/reinforcement_learning/SAC_tensorboard/reach_3"


def sanitize_filename(name):
    # Replace or remove characters that aren't valid in filenames
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)


def plot_and_save_tensorboard_scalars(log_dir):
    """
    Parses TensorBoard event files and saves each scalar as a plot with filenames
    prefixed by `filename_prefix`.
    """
    log_dir = os.path.expanduser(log_dir)

    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory {log_dir} not found.")

    os.makedirs(log_dir, exist_ok=True)

    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()

    tags = accumulator.Tags().get('scalars', [])
    if not tags:
        print(f"No scalar data found in {log_dir}.")
        return

    for tag in tags:
        events = accumulator.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, values, label=tag)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(f'Scalar: {tag}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        sanitized_tag = sanitize_filename(tag)
        filename = f"{sanitized_tag}.pdf"
        filepath = os.path.join(log_dir, filename)

        plt.savefig(filepath)
        plt.close()
        print(f"Saved plot: {filepath}")


if __name__ == "__main__":
    plot_and_save_tensorboard_scalars(TENSOR_FOLDER)
