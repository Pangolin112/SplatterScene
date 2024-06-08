import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the depth map image
image_path = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/depth/DSC00350.jpg'
depth_map = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Normalize the depth map to the range 0-255
normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

# Apply a color map to the normalized depth map
heatmap = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_JET)

# Display the heat map using matplotlib
plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
plt.title('Heat Map')
plt.axis('off')
plt.show()

# Save the heatmap image
output_path = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/gt_depth_heatmap/heatmap_DSC00350.jpg'
cv2.imwrite(output_path, heatmap)

output_path
