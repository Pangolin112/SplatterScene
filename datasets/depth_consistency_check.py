from PIL import Image
import numpy as np

# Load images
img1_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/0cf2e9402d/depth_from_gt/debug/DSC00351.jpg'
img2_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/0cf2e9402d/depth/DSC00351.jpg'

img1 = Image.open(img1_path).convert('L')
img2 = Image.open(img2_path).convert('L')

# Convert images to numpy arrays
img1_array = np.array(img1)
img2_array = np.array(img2)

# Define the region of interest
roi = (0, 0, 128, 128)

# Crop the regions
img1_roi = img1_array[roi[0]:roi[2], roi[1]:roi[3]]
img2_roi = img2_array[roi[0]:roi[2], roi[1]:roi[3]]

# Mask out pixels with value 0 or 255
mask1 = (img1_roi != 0) & (img1_roi != 255)
mask2 = (img2_roi != 0) & (img2_roi != 255)

masked_img1_roi = np.where(mask1, img1_roi, np.nan)
masked_img1_roi = np.where(mask2, masked_img1_roi, np.nan)
masked_img2_roi = np.where(mask2, img2_roi, np.nan)
masked_img2_roi = np.where(mask1, masked_img2_roi, np.nan)

# Compute L2 loss (Euclidean distance)
l1_loss = np.nansum(np.abs(masked_img1_roi - masked_img2_roi)) / (128*128)
l2_loss = np.nansum((masked_img1_roi - masked_img2_roi) ** 2) / (128*128)

# Compute statistics
mean1 = np.nanmean(masked_img1_roi)
mean2 = np.nanmean(masked_img2_roi)

max1 = np.nanmax(masked_img1_roi)
max2 = np.nanmax(masked_img2_roi)

min1 = np.nanmin(masked_img1_roi)
min2 = np.nanmin(masked_img2_roi)

std1 = np.nanstd(masked_img1_roi)
std2 = np.nanstd(masked_img2_roi)

pixel_differences = masked_img1_roi - masked_img2_roi
max_difference = np.nanmax(pixel_differences)
print(f"L1 Loss: {l1_loss}")
print(f"L2 Loss: {l2_loss}")
print(f"Mean Pixel Value - Image 1: {mean1}, Image 2: {mean2}")
print(f"Maximum Pixel Value - Image 1: {max1}, Image 2: {max2}")
print(f"Minimum Pixel Value - Image 1: {min1}, Image 2: {min2}")
print(f"Standard Deviation - Image 1: {std1}, Image 2: {std2}")
print(f"Maximum Pixel Difference: {max_difference}")
print("Pixel Differences:")
print(pixel_differences)
