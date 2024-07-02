from PIL import Image
import os
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import maximum_filter

# Define the source and destination folders
scene_name = '0a7cc12c0e'
source_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/{scene_name}/render_depth'
destination_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/{scene_name}/depth'

for file_name in os.listdir(source_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Check for image files
        # Open the image
        img_path = os.path.join(source_folder, file_name)
        img = Image.open(img_path)

        # Convert the image to 'F' mode (32-bit floating point) for resizing
        img = img.convert('F')

        # Resize the image
        img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)

        # Convert to numpy array for scaling
        img_array = np.array(img_resized)

        img_array[img_array < 0] = 0

        # Normalize the pixel values to fit within the 8-bit range (0-255)
        img_array_normalized = (img_array - 0) / (10000.0 - 0) * 255.0
        img_array_normalized = img_array_normalized.astype(np.uint8)

        # Set pixels with values less than 10 to zero
        img_array_normalized[img_array_normalized == 0] = 0

        # Pad the array to handle the boundary conditions
        padded_img = np.pad(img_array_normalized, pad_width=5, mode='constant', constant_values=0)

        # Apply the maximum filter with a size of 11x11 (5 pixels in each direction)
        max_filtered = maximum_filter(padded_img, size=11)

        # Update only the zero values in the original image within the boundary
        for i in range(128):
            for j in range(128):
                if img_array_normalized[i][j] == 0:
                    img_array_normalized[i][j] = max_filtered[i][j]


        for i in range(128):
            for j in range(128):
                if img_array_normalized[i][j] == 0 and j > 0:
                    img_array_normalized[i][j] = img_array_normalized[i][j - 1]

        for i in range(128):
            for j in range(128):
                if img_array_normalized[i][127 - j] == 0 and j > 0:
                    img_array_normalized[i][127 - j] = img_array_normalized[i][127 - j + 1]

        # Convert the normalized array back to an image in 'L' mode (8-bit pixels, black and white)
        img_normalized = Image.fromarray(img_array_normalized, mode='L')

        # Save the resized image to the destination folder as JPEG
        jpeg_file_name = os.path.splitext(file_name)[0] + '.jpg'
        img_normalized.save(os.path.join(destination_folder, jpeg_file_name), format='JPEG')

print("Image resized and saved successfully.")
