import json

name_scene = '49a82360aa'
base_path = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/{name_scene}/'

# Load the JSON data from file
with open(base_path + 'nerfstudio/transforms_undistorted.json', 'r') as file:
    data = json.load(file)

image_path = base_path + "undistorted_images/"

# Loop over each frame in the JSON data
for frame in data['frames']:
    file_path = image_path + frame['file_path']
    transform_matrix = frame['transform_matrix']

    # Create a filename for the .txt file corresponding to the image
    txt_filename = frame['file_path'] + '.txt'
    txt_filepath = base_path + 'pose_nerfstudio/' + txt_filename

    # Write the transform matrix to the .txt file
    with open(txt_filepath, 'w') as txt_file:
        #for row in transform_matrix:
            #txt_file.write(' '.join(f"{num:.6f}" for num in row) + '\n')
        single_line = ' '.join(f"{num:.6f}" for row in transform_matrix for num in row)
        txt_file.write(single_line + '\n')

for frame in data['test_frames']:
    file_path = image_path + frame['file_path']
    transform_matrix = frame['transform_matrix']

    # Create a filename for the .txt file corresponding to the image
    txt_filename = frame['file_path'] + '.txt'
    txt_filepath = base_path + 'pose_nerfstudio/' + txt_filename

    # Write the transform matrix to the .txt file
    with open(txt_filepath, 'w') as txt_file:
        #for row in transform_matrix:
            #txt_file.write(' '.join(f"{num:.6f}" for num in row) + '\n')
        single_line = ' '.join(f"{num:.6f}" for row in transform_matrix for num in row)
        txt_file.write(single_line + '\n')

print("Extraction complete. Files are saved with corresponding names.")
