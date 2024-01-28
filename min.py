import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

def calculate_percentage(mask, total_pixels):
    return np.sum(mask) / total_pixels * 100

def load_image(image_path):
    return io.imread(image_path)

def process_image(image, sat_threshold, hue_threshold):
    # Convert to HSV
    hsv_image = color.rgb2hsv(image)

    # Apply thresholds
    above_sat_mask = hsv_image[:, :, 1] > sat_threshold
    below_sat_mask = hsv_image[:, :, 1] <= sat_threshold
    cluster1_mask = np.logical_and(above_sat_mask, hsv_image[:, :, 0] < hue_threshold)
    cluster2_mask = np.logical_and(above_sat_mask, hsv_image[:, :, 0] >= hue_threshold)
    
    # Calculate percentages
    total_pixels = image.shape[0] * image.shape[1]
    calchopyrite_percent = calculate_percentage(cluster1_mask, total_pixels)
    other_sulphides_percent = calculate_percentage(cluster2_mask, total_pixels)
    non_sulphides_percent = calculate_percentage(below_sat_mask, total_pixels)

    return calchopyrite_percent, other_sulphides_percent, non_sulphides_percent

def display_images(image, cluster1_mask, cluster2_mask, cluster3_mask):
    # Create images for each cluster
    cluster_images = []
    for mask in [cluster1_mask, cluster2_mask, cluster3_mask]:
        cluster_image = image.copy()
        cluster_image[~mask] = [0, 0, 0]
        cluster_images.append(cluster_image)
    
    # Plotting
    fig, ax = plt.subplots(4, 1, figsize=(6, 12))
    ax[0].imshow(image)
    ax[0].axis('off')
    for i, cluster_image in enumerate(cluster_images):
        ax[i+1].imshow(cluster_image)
        ax[i+1].axis('off')
    st.pyplot(fig)

def main():
    st.title("Mineral Image Analysis")
    
    # Default thresholds
    default_sat_threshold = 0.1
    default_hue_threshold = 0.3

    # User input for thresholds
    sat_threshold = st.sidebar.slider("Saturation Threshold", 0.0, 1.0, default_sat_threshold)
    hue_threshold = st.sidebar.slider("Hue Threshold", 0.0, 1.0, default_hue_threshold)

    # Image selection
    selected_image = st.selectbox("Select an Image", ['calco.jpg', 'other_images'])

    if selected_image:
        image = load_image(selected_image)
        calchopyrite_percent, other_sulphides_percent, non_sulphides_percent = process_image(image, sat_threshold, hue_threshold)
        
        # Display results
        st.write(f"Calchopyrite: {calchopyrite_percent:.2f}%")
        st.write(f"Other Sulphides: {other_sulphides_percent:.2f}%")
        st.write(f"Non Sulphides: {non_sulphides_percent:.2f}%")
        
        display_images(image, cluster1_mask, cluster2_mask, cluster3_mask)

if __name__ == "__main__":
    main()
