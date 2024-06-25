import os
import shutil


def delete_files_in_folder(folder_path):
    """
    Delete all files in the specified folder.

    :param folder_path: Path to the folder where files should be deleted
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def move_and_overwrite_files(index_list):
    """
    Move selected files from source_folder to destination_folder, overwriting any existing files.

    :param source_folder: Path to the source folder
    :param destination_folder: Path to the destination folder
    :param start_index: Starting index of the files to move (inclusive)
    :param end_index: Ending index of the files to move (inclusive)
    """
    for k in range(0, 3):
        if k == 0:
            folder_type = 'train'
        elif k == 1:
            folder_type = 'val'
        else:
            folder_type = 'test'
        for i in index_list:
            for j in range(0, 4):
                if j == 0:
                    folder_name = 'rgb'
                    file_name = f"DSC{i:05d}.JPG"
                elif j == 1:
                    folder_name = 'depth'
                    file_name = f"DSC{i:05d}.jpg"
                elif j == 2:
                    folder_name = 'pose_colmap_depth'
                    file_name = f"DSC{i:05d}.JPG.txt"
                else:
                    folder_name = 'pose_nerfstudio_rgb'
                    file_name = f"DSC{i:05d}.JPG.txt"

                source_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/0cf2e9402d/{folder_name}/'
                destination_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/srn_cars/cars_{folder_type}/0cf2e9402d_train/{folder_name}/'

                # Adjust the formatting according to your file naming convention
                source_file = os.path.join(source_folder, file_name)
                destination_file = os.path.join(destination_folder, file_name)

                # Check if the source file exists
                if os.path.exists(source_file):
                    # Move the file, overwriting the destination file if it exists
                    shutil.copy(source_file, destination_file)
                    print(f"Copied {file_name} to {destination_folder}")
                else:
                    print(f"File {file_name} does not exist in {source_folder}")


# Get the file range from the user
# start_index = int(input("Enter the starting index of the files to move: "))
# end_index = int(input("Enter the ending index of the files to move: "))

# delete files before copying
for q in range(0, 3):
    if q == 0:
        folder_type = 'train'
    elif q == 1:
        folder_type = 'val'
    else:
        folder_type = 'test'
    for p in range(0, 4):
        if p == 0:
            folder_name = 'rgb'
        elif p == 1:
            folder_name = 'depth'
        elif p == 2:
            folder_name = 'pose_colmap_depth'
        else:
            folder_name = 'pose_nerfstudio_rgb'

        destination_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/srn_cars/cars_{folder_type}/0cf2e9402d_train/{folder_name}/'
        delete_files_in_folder(destination_folder)

# index_list = [i for i in range(101, 109)]
index_list = [350,355]

move_and_overwrite_files(index_list)
