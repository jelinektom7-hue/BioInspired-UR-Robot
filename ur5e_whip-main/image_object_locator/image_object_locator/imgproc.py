import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def load_image(path, color_space=['bgr']):
    imgs = []
    for cs in color_space:
        if cs == 'bgr':
            bgr = cv.imread(path)
            imgs.append(bgr)
        elif cs == 'rgb':
            rgb = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
            imgs.append(rgb)
        elif cs == 'lab':
            lab = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2Lab)
            imgs.append(lab)
        else:
            raise ValueError(f"Unsupported color space: {cs}")
    return imgs


def process_annotated_images(folder_path):
    annotated_images = [f for f in os.listdir(folder_path) if 'annotated' in f]
    original_images = [f for f in os.listdir(folder_path) if 'original' in f]

    color_data_list_rgb = []

    for annotated_image in annotated_images:
        original_image = annotated_image.replace("annotated", "original")
        
        if original_image in original_images:
            path_image_anno = os.path.join(folder_path, annotated_image)
            path_image_orig = os.path.join(folder_path, original_image)
            
            orig_img = load_image(path_image_orig, color_space=['rgb'])[0]
            anno_img = load_image(path_image_anno, color_space=['rgb'])[0]
            anno_color_rgb = np.array([0, 0, 255], dtype=np.uint8)
            
            # Get annotated color data (mask, pixels_selected_by_anno, mean, std)
            color_data = get_annotated_color_data(orig_img, anno_img, anno_color_rgb)
            color_data_list_rgb.append(color_data)

    return color_data_list_rgb


def get_annotated_color_data(orig_img, anno_img, anno_color):
    mask = cv.inRange(anno_img, anno_color, anno_color)
    
    pixels_selected_by_anno = orig_img[mask > 0]
    
    mean = np.mean(pixels_selected_by_anno, axis=0)
    std = np.std(pixels_selected_by_anno, axis=0)
    
    return mask, pixels_selected_by_anno, mean, std


def plot_color_distribution(image_masked_no_zeros, color_space, show=True):
    """Plot the histograms of the RGB color channels."""
    
    # Split the channels
    channels = []
    colors = []
    labels = []
    if color_space == "BGR":
        r_channel = image_masked_no_zeros[:, 2]  # Red channel
        g_channel = image_masked_no_zeros[:, 1]  # Green channel
        b_channel = image_masked_no_zeros[:, 0]  # Blue channel
        channels = [r_channel, g_channel, b_channel]
        colors = ['red', 'green', 'blue']
        labels = ['Red', 'Green', 'Blue']
    if color_space == "RGB":
        r_channel = image_masked_no_zeros[:, 0]  # Red channel
        g_channel = image_masked_no_zeros[:, 1]  # Green channel
        b_channel = image_masked_no_zeros[:, 2]  # Blue channel
        channels = [r_channel, g_channel, b_channel]
        colors = ['red', 'green', 'blue']
        labels = ['Red', 'Green', 'Blue']
    elif color_space == "Lab":
        l_channel = image_masked_no_zeros[:, 0]  # L channel
        a_channel = image_masked_no_zeros[:, 1]  # a channel
        b_channel = image_masked_no_zeros[:, 2]  # b channel
        channels = [l_channel, a_channel, b_channel]
        colors = ['gray', 'purple', 'orange']
        labels = ['L', 'a', 'b']
    else:
        raise ValueError("Invalid color space. Use 'RGB' or 'Lab'.")

    # Create the histogram
    plt.figure(figsize=(10, 5))

    # Plot histograms for each channel
    for channel, color, label in zip(channels, colors, labels):
        hist, bins = np.histogram(channel.ravel(), bins=256)
        max_bin_index = np.argmax(hist)
        hist[max_bin_index] = 0  # Remove the bin with the highest frequency
        plt.bar(bins[:-1], hist, color=color, alpha=0.6, label=label, width=1)

    # Labels and title
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title(color_space + ' Color Histogram')
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def segment_image_by_color_inrange(image, color, tolerance_in_percent, color_space, method="inRange"):
    lower_bound = None
    upper_bound = None
    
    if color_space == "RGB":
        tolerance = 255 * tolerance_in_percent / 100
        lower_bound = color - tolerance
        upper_bound = color + tolerance
    elif color_space == "Lab":
        tolerance = (255 * tolerance_in_percent / 100)
        
        l_tolerance = tolerance * (100 / 255)  # L channel ranges from 0 to 100
        ab_tolerance = tolerance * (128 / 255)  # a/b channels range from -128 to 127
        
        lower_bound = np.array([color[0] - l_tolerance, color[1] - ab_tolerance, color[2] - ab_tolerance])
        upper_bound = np.array([color[0] + l_tolerance, color[1] + ab_tolerance, color[2] + ab_tolerance])
    else:
        raise ValueError("Invalid color space. Use 'RGB' or 'Lab'.")
    
    return cv.inRange(image, lower_bound, upper_bound)


def segment_image_by_color_distance(image, reference_color, distance_image, threshold, method="Euclidean"):
    pixels = np.reshape(image, (-1, 3))
    shape = pixels.shape
    if method == "Euclidean":
        # segment pumpkins in RGB space using Euclidean distance and Mahalanobis distance to reference color
        diff = pixels - np.repeat([reference_color], shape[0], axis=0)
        euclidean_dist = np.sqrt(np.sum(diff * diff, axis=1))
        euclidean_dist_image = np.reshape(euclidean_dist, 
                (image.shape[0], image.shape[1]))

        euclidean_dist_image_scaled = 255 * euclidean_dist_image / np.max(euclidean_dist_image)
        euclidean_dist_image_scaled = euclidean_dist_image_scaled.astype(np.uint8)
        
        segmented_image_euclidean = cv.inRange(euclidean_dist_image_scaled, threshold, 255)
        return np.max(segmented_image_euclidean) - segmented_image_euclidean
    elif method == "Mahalanobis":
        covariance_matrix = np.cov(pixels, rowvar=False)
        diff = pixels - np.repeat([reference_color], shape[0], axis=0)
        inv_cov = np.linalg.pinv(covariance_matrix)
        moddotproduct = diff * (diff @ inv_cov)
        mahalanobis_dist = np.sum(moddotproduct, 
            axis=1)
        mahalanobis_distance_image = np.reshape(
            mahalanobis_dist, 
                (image.shape[0],
                image.shape[1]))

        # Scale the distance image and export it.
        if np.max(mahalanobis_distance_image) > 0:
            mahalanobis_distance_image = 255 * mahalanobis_distance_image / np.max(mahalanobis_distance_image)
        else:
            print("Warning: Mahalanobis distance is zero everywhere.")
        mahalanobis_distance_image = mahalanobis_distance_image.astype(np.uint8)
        segmented_image_mahalanobis = cv.inRange(mahalanobis_distance_image, threshold, 255)
        return np.max(segmented_image_mahalanobis) - segmented_image_mahalanobis
    else:
        raise ValueError("Invalid method. Use 'Euclidean' or 'Mahalanobis'.")


def compare_original_and_segmented_image(fig, original, segmented, title, subplot_idx):
    """
    Adds the original and segmented image pair to the given figure as a subplot.

    Args:
        fig: The matplotlib figure to add the subplots to.
        original: The original image (RGB or other).
        segmented: The segmented image (RGB or other).
        title: The title for the subplot.
        subplot_idx: The index for the subplot position.
    """
    # Check if the subplot_idx is within the range of available slots (1 to 8 for a 4x2 grid)
    if subplot_idx > 8:
        raise ValueError("Cannot have more than 8 subplot pairs in a 4x2 grid.")
    
    # Create the subplot for the original image
    ax = fig.add_subplot(4, 2, subplot_idx)  # 4 rows, 2 columns grid
    ax.set_title(f"{title} - Original")
    ax.imshow(original)
    ax.axis('off')  # Turn off axis for a cleaner view
    
    # Create the subplot for the segmented image in the next position
    next_subplot_idx = subplot_idx + 1
    if next_subplot_idx > 8:
        raise ValueError("Cannot have more than 8 subplot pairs in a 4x2 grid.")
    
    ax2 = fig.add_subplot(4, 2, next_subplot_idx)  # Ensure the next slot for the segmented image
    ax2.set_title(f"{title} - Segmented")
    ax2.imshow(segmented)
    ax2.axis('off')  # Turn off axis for a cleaner view

    # Adjust layout for spacing between subplots
    plt.tight_layout()