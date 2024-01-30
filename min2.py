import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Function to load an image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Function to perform K-means clustering
def segment_image_kmeans(image, n_clusters, color_space):
    # Convert image to the selected color space
    if color_space == 'RGB':
        pass  # Already in RGB
    elif color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(pixel_values)

    # Convert labels to image
    labels = labels.reshape(image.shape[:2])
    return labels

# Function to perform PCA + K-means clustering
def segment_image_pca_kmeans(image, n_clusters, n_components):
    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Apply PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(pixel_values)

    # Apply KMeans clustering on PCA-transformed data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(transformed)

    # Convert labels to image
    labels = labels.reshape(image.shape[:2])
    return labels

# Function to display the original and segmented images
def display_segmentation(image, labels):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(labels, cmap='viridis')
    plt.title('Segmented Image')
    plt.axis('off')
    st.pyplot(plt.gcf())

# Streamlit main function
def main():
    st.title("Mineral Image Analysis")

    tab1, tab2, tab3 = st.tabs(["Manual Segmentation", "K-Means Segmentation", "PCA + K-Means Segmentation"])

    with tab1:
        st.write("Manual segmentation options...")
        # Your existing manual segmentation code here

    with tab2:
        st.write("K-Means segmentation options...")
        selected_image_kmeans = st.selectbox("Select an Image for K-Means", ['calco.jpg', 'other_images'], key='image_select_kmeans')
        n_clusters_kmeans = st.slider("Number of Clusters", 2, 10, 3, key='n_clusters_kmeans')
        color_space_kmeans = st.selectbox("Color Space", ['RGB', 'HSV', 'LAB'], key='color_space_kmeans')

        if st.button('Segment Image with K-Means'):
            if selected_image_kmeans:
                image_kmeans = load_image(selected_image_kmeans)
                labels_kmeans = segment_image_kmeans(image_kmeans, n_clusters_kmeans, color_space_kmeans)
                display_segmentation(image_kmeans, labels_kmeans)

    with tab3:
        st.write("PCA + K-Means segmentation options...")
        selected_image_pca = st.selectbox("Select an Image for PCA + K-Means", ['calco.jpg', 'other_images'], key='image_select_pca')
        n_clusters_pca = st.slider("Number of Clusters", 2, 10, 3, key='n_clusters_pca')
        n_components_pca = st.slider("Number of PCA Components", 1, 3, 2, key='n_components_pca')

        if st.button('Segment Image with PCA + K-Means'):
            if selected_image_pca:
                image_pca = load_image(selected_image_pca)
                labels_pca = segment_image_pca_kmeans(image_pca, n_clusters_pca, n_components_pca)
                display_segmentation(image_pca, labels_pca)

if __name__ == "__main__":
    main()
