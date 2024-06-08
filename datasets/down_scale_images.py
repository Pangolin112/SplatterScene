from PIL import Image
import os
import numpy as np

# Define the source and destination folders
source_folder = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/0cf2e9402d/render_depth_original_size'
destination_folder = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/0cf2e9402d/depth'

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

        # Normalize the pixel values to fit within the 8-bit range (0-255)
        img_array_normalized = (img_array - img_array.min()) / (10000.0 - img_array.min()) * 255.0
        img_array_normalized = img_array_normalized.astype(np.uint8)

        # Convert the normalized array back to an image in 'L' mode (8-bit pixels, black and white)
        img_normalized = Image.fromarray(img_array_normalized, mode='L')

        # Save the resized image to the destination folder as JPEG
        jpeg_file_name = os.path.splitext(file_name)[0] + '.jpg'
        img_normalized.save(os.path.join(destination_folder, jpeg_file_name), format='JPEG')

        '''
        # Scale the pixel values to fit within the 16-bit range (0-65535)
        #img_array_scaled = (img_array * (65535.0 / img_array.max())).astype(np.uint16)
        img_array_scaled = img_array.astype(np.uint16)

        # Convert the scaled array back to an image in 'I;16' mode
        img_rescaled = Image.fromarray(img_array_scaled, mode='I;16')

        # Save the resized image to the destination folder
        img_rescaled.save(os.path.join(destination_folder, file_name))'''

print("Image resized and saved successfully.")
