import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
import os
import gymnasium_env
import csv
import numpy as np
import matplotlib.pyplot as plt


ENVIRONMENT = 'gymnasium_env/WhipWorld-v0'
MODEL_FILE = os.path.expanduser("~") + "/ros2_ws/src/ur5e_whip/reinforcement_learning/SAC_models_09_reach_05/rl_model_3070000_steps.zip"
SAVE_LOCATION = os.path.expanduser("~") + "/ros2_ws/src/ur5e_whip/reinforcement_learning/SAC_recordings"


def prune_to_size(csv_file: str, target_points: int):
    # Load CSV
    timestamps = []
    joint_data = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        num_joints = len(header) - 1

        for row in reader:
            joint_values = list(map(float, row[:num_joints]))
            elapsed_time = float(row[-1])
            joint_data.append(joint_values)
            timestamps.append(elapsed_time)

    joint_data = np.array(joint_data)
    timestamps = np.array(timestamps)

    total_points = len(timestamps)

    if target_points >= total_points:
        print(f"Target number of points ({target_points}) is greater than or equal to total data points ({total_points}). No pruning needed.")
        return csv_file

    # --- Downsampling: uniformly select indices ---
    indices = np.linspace(0, total_points - 1, target_points, dtype=int)

    pruned_joint_data = joint_data[indices]
    pruned_timestamps = timestamps[indices]

    # --- Save pruned data ---
    base, ext = os.path.splitext(csv_file)
    pruned_file = base + "_pruned" + ext

    with open(pruned_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"joint_{i+1}" for i in range(num_joints)] + ["elapsed_time [s]"])
        for row, t in zip(pruned_joint_data, pruned_timestamps):
            writer.writerow(row.tolist() + [t])

    print(f"Pruned data saved to: {pruned_file}")
    return pruned_file



def prune(csv_file: csv, tolerance = 0.001, ):
    # Load CSV
    
    timestamps = []
    joint_data = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        num_joints = len(header) - 1

        for row in reader:
            joint_values = list(map(float, row[:num_joints]))
            elapsed_time = float(row[-1])
            joint_data.append(joint_values)
            timestamps.append(elapsed_time)

    joint_data = np.array(joint_data)
    timestamps = np.array(timestamps)

    # --- Pruning Function ---
    def prune_data(joint_data, timestamps, tolerance=0.01):
        pruned_joint_data = [joint_data[0]]
        pruned_timestamps = [timestamps[0]]

        for i in range(1, len(joint_data) - 1):
            diff_prev = np.abs(joint_data[i] - joint_data[i - 1])
            diff_next = np.abs(joint_data[i] - joint_data[i + 1])
            max_diff = np.maximum(diff_prev, diff_next).max()

            if max_diff > tolerance:
                pruned_joint_data.append(joint_data[i])
                pruned_timestamps.append(timestamps[i])

        pruned_joint_data.append(joint_data[-1])
        pruned_timestamps.append(timestamps[-1])

        return np.array(pruned_joint_data), np.array(pruned_timestamps)

    # Prune
    pruned_joint_data, pruned_timestamps = prune_data(joint_data, timestamps, tolerance=tolerance)

    # --- Save pruned data ---
    # Construct new filename
    base, ext = os.path.splitext(csv_file)
    pruned_file = base + "_pruned" + ext

    with open(pruned_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"joint_{i+1}" for i in range(num_joints)] + ["elapsed_time [s]"])
        for row, t in zip(pruned_joint_data, pruned_timestamps):
            writer.writerow(row.tolist() + [t])

    print(f"Pruned data saved to: {pruned_file}")
    return pruned_file


def plot(csv_file: str, pruned_file: str):
    # === Load origional Data ===
    timestamps = []
    joint_data = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        num_joints = len(header) - 1

        for row in reader:
            joint_values = list(map(float, row[:num_joints]))
            elapsed_time = float(row[-1])
            joint_data.append(joint_values)
            timestamps.append(elapsed_time)

    joint_data = np.array(joint_data)
    timestamps = np.array(timestamps)

    # === Load Pruned Data ===
    pruned_joint_data = []
    pruned_timestamps = []

    with open(pruned_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            joint_values = list(map(float, row[:num_joints]))
            elapsed_time = float(row[-1])
            pruned_joint_data.append(joint_values)
            pruned_timestamps.append(elapsed_time)

    pruned_joint_data = np.array(pruned_joint_data)
    pruned_timestamps = np.array(pruned_timestamps)

    # === Plot ===
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10', num_joints)

    for i in range(num_joints):
        color = colors(i)
        plt.plot(timestamps, joint_data[:, i], label=f'Joint {i+1} (original)', color=color, alpha=0.4)
        plt.plot(pruned_timestamps, pruned_joint_data[:, i], label=f'Joint {i+1} (pruned)', color=color, linestyle='--', linewidth=2)

    plt.xlabel('Elapsed Time [s]')
    plt.ylabel('Joint Value')
    plt.title('Joint Values: Original vs Pruned (Loaded from Files)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def record_session(include_timestamp = False, include_header = False) -> str:
    # Load environment
    env = gym.make(ENVIRONMENT, render_mode="human")
    model = SAC.load(MODEL_FILE)

    while True:
        obs, info = env.reset()
        done = False
        terminated = False
        total_reward = 0

        episode_data = []

        print("New episode")
        while not done:
            action, info = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Extract joint values and timestamp
            joint_values = obs['agent_joint_values']  # Replace with correct key from your obs
            elapsed = info["elapsed time [s]"]

            # Save data as a row
            episode_data.append([joint_values.tolist(), elapsed])
        

        print(f"Total Reward: {total_reward:.2f}")

        x = input("Press \"s\" to save session. Press \"a\" to try again. Press \"e\" to exit.\n")

        if x == "e" or x == "E":
            env.close()
            exit()

        if x == "s" or x == "S":
            save_file = input("Give a name to the saved run: ")
            csv_filename = SAVE_LOCATION + "/" + save_file + ".csv"

            # Print target position
            print("target_position: x=" + str(obs["target_position"][0]) + ", y=" + str(obs["target_position"][1]) + ", z=" + str(obs["target_position"][2]))

            # Save episode data to CSV
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                if include_header:
                    if include_timestamp:
                        writer.writerow([f"joint_{i+1}" for i in range(len(episode_data[0][0]))] + ["elapsed_time [s]"])
                    else:
                        writer.writerow([f"joint_{i+1}" for i in range(len(episode_data[0][0]))])

                for joint_values, elapsed in episode_data:
                    if include_timestamp:
                        writer.writerow(joint_values + [elapsed])
                    else:
                        writer.writerow(joint_values)
            
            print(f"Data saved to: {csv_filename}")
            
            env.close()
            return csv_filename
            

    


def main():
    # data_file = record_session()
    data_file = "/home/anders/ros2_ws/src/ur5e_whip/reinforcement_learning/SAC_recordings/god.csv"
    #pruned_data_file = prune(data_file)
    pruned_data_file = prune_to_size(data_file, target_points=100)
    plot(data_file, pruned_data_file)


if __name__ == "__main__":
    main()

# god: x=0.4511656631696203, y=-0.08818698171233313, z=0.9436654577163347