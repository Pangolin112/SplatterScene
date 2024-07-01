import os
import shutil
import base_path as bp

base_path = bp.base_path

def delete_files_in_folder(folder_path):
    """
    Delete all files in the specified folder.

    :param folder_path: Path to the folder where files should be deleted
    """
    if not os.path.exists(folder_path):
        # Create the folder if it does not exist
        os.makedirs(folder_path)

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


def move_and_overwrite_files(scene_name, index_list, folder_type):
    """
    Move selected files from source_folder to destination_folder, overwriting any existing files.

    :param source_folder: Path to the source folder
    :param destination_folder: Path to the destination folder
    :param start_index: Starting index of the files to move (inclusive)
    :param end_index: Ending index of the files to move (inclusive)
    """
    for i in index_list:
        for j in range(0, 3):
            if j == 0:
                folder_name = 'rgb'
                file_name = f"DSC{i:05d}.JPG"
            elif j == 1:
                folder_name = 'depth'
                file_name = f"DSC{i:05d}.jpg"
            else:
                folder_name = 'pose_colmap_depth'
                file_name = f"DSC{i:05d}.JPG.txt"

            source_folder = base_path + f'{scene_name}/{folder_name}/'
            destination_folder = base_path + f'srn_cars/cars_{folder_type}/{scene_name}_{folder_type}/{folder_name}/'

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


scene_name = '0cf2e9402d'
index_list_train = [350, 352, 354, 356, 358, 360, 362, 364, 366]
index_list_val = [353, 357, 361, 365]
index_list_test = [351, 355, 359, 363]
# index_list_train = [350, 352, 354, 356, 358, 360, 362, 364, 366]
# index_list_val = [353, 357, 361, 363, 365]
# index_list_test = [351, 355, 359, 363, 365]
# index_list_train = [350, 358, 366]
# index_list_val = [350, 358, 366]
# index_list_test = [350, 358, 366]
# index_list_train = [350, 366]
# index_list_val = [350, 366]
# index_list_test = [350, 366]

# scene_name = '49a82360aa'
# index_list_train = [131, 133, 135, 137, 141, 143, 145]
# index_list_val = [134, 138, 139, 142]
# index_list_test = [132, 136, 140, 144]

index_dict = {
    'train': index_list_train,
    'val': index_list_val,
    'test': index_list_test
}

for folder_type, index_list in index_dict.items():
    for p in range(0, 3):
        if p == 0:
            folder_name = 'rgb'
        elif p == 1:
            folder_name = 'depth'
        else:
            folder_name = 'pose_colmap_depth'

        destination_folder = base_path + f'srn_cars/cars_{folder_type}/{scene_name}_{folder_type}/{folder_name}/'
        delete_files_in_folder(destination_folder)
        move_and_overwrite_files(scene_name, index_list, folder_type)


