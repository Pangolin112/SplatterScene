from PIL import Image
import os
import numpy as np
from scipy.ndimage import distance_transform_edt

# Define the source and destination folders
scene_name = '0a7cc12c0e'
source_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/{scene_name}/images'
destination_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/{scene_name}/rgb'

for file_name in os.listdir(source_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Check for image files
        # Open the image
        img_path = os.path.join(source_folder, file_name)
        img = Image.open(img_path)

        # Convert the image to 'F' mode (32-bit floating point) for resizing
        # img = img.convert('F')

        # Resize the image
        img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)

        # Save the resized image to the destination folder as JPEG
        jpeg_file_name = os.path.splitext(file_name)[0] + '.JPG'
        img_resized.save(os.path.join(destination_folder, jpeg_file_name), format='JPEG')

        '''
        # Scale the pixel values to fit within the 16-bit range (0-65535)
        #img_array_scaled = (img_array * (65535.0 / img_array.max())).astype(np.uint16)
        img_array_scaled = img_array.astype(np.uint16)

        # Convert the scaled array back to an image in 'I;16' mode
        img_rescaled = Image.fromarray(img_array_scaled, mode='I;16')

        # Save the resized image to the destination folder
        img_rescaled.save(os.path.join(destination_folder, file_name))'''



print("Image resized and saved successfully.")
