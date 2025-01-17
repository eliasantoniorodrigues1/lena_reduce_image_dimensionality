import cv2
import numpy as np
from sklearn.decomposition import PCA
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def reduce_image_dimensionality(image_path, n_components=50):
    """
    Reduces the dimensionality of an image using PCA.

    Args:
        image_path: Path to the image file.
        n_components: Number of principal components to retain.

    Returns:
        A NumPy array representing the reconstructed image with reduced 
        dimensionality,
        or None if the image cannot be loaded.
    """
    try:
        # Load the image using OpenCV
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return None

        # Convert the image to grayscale (optional, but often helpful for
        # dimensionality reduction)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Reshape the image into a matrix of pixels
        pixels = gray_img.reshape(-1, 1)  # Each pixel becomes a row

        # Ensure n_components is valid
        n_samples, n_features = pixels.shape
        n_components = min(n_components, n_samples, n_features)

        pca = PCA(n_components=n_components)
        pca.fit(pixels)
        transformed_pixels = pca.transform(pixels)
        reconstructed_pixels = pca.inverse_transform(transformed_pixels)

        # Reshape back into image dimensions
        reconstructed_img = reconstructed_pixels.reshape(gray_img.shape)

        # Convert back to uint8 (if needed)
        reconstructed_img = reconstructed_img.astype(np.uint8)

        return reconstructed_img
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == '__main__':
    # Example usage:  Replace with your image path
    file_name = 'lena'
    image_path = os.path.join(BASE_DIR, 'image', f'{file_name}.jpg')
    reconstructed_image = reduce_image_dimensionality(image_path)

    if reconstructed_image is not None:
        # Display the original and reconstructed images (optional)
        cv2.imshow('Original Image', cv2.imread(image_path))
        cv2.imshow('Reconstructed Image', reconstructed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the reconstructed image (optional)
        cv2.imwrite(os.path.join(BASE_DIR, 'image',
                    'reconstructed_{file_name}.jpg'), reconstructed_image)
        
        print('Reconstructed image saved successfully.')
