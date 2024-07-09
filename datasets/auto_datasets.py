import os
import shutil

import base_path as bp

base_path = bp.srn1_path

def delete_files_in_folder(folder_path):
    """
    Delete all files in the specified folder.

    :param folder_path: Path to the folder where files should be deleted
    """
    if not os.path.exists(folder_path):
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

    :param scene_name: Name of the scene
    :param index_list: List of file indices to move
    :param folder_type: Type of folder (train, val, test)
    """
    for i in index_list:
        for j in range(0, 3):
            if j == 0:
                folder_name = 'rgb'
                file_name = f"DSC{i:05d}.JPG"
            elif j == 1:
                folder_name = 'depth'
                # file_name = f"DSC{i:05d}.jpg"
                file_name = f"DSC{i:05d}.png"
            else:
                folder_name = 'pose_colmap_depth'
                file_name = f"DSC{i:05d}.JPG.txt"

            source_folder = os.path.join(base_path, scene_name, folder_name)
            destination_folder = os.path.join(base_path, 'srn_cars', f'cars_{folder_type}', f'{scene_name}', folder_name)

            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)

            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            if os.path.exists(source_file):
                try:
                    shutil.copy(source_file, destination_file)
                    print(f"Copied {file_name} to {destination_folder}")
                except Exception as e:
                    print(f"Failed to copy {source_file} to {destination_file}. Reason: {e}")
            else:
                print(f"File {file_name} does not exist in {source_folder}")

# scene_name = '0cf2e9402d'
# index_list_train = [350, 352, 354, 356, 358, 360, 362, 364, 366, 351, 355, 359, 363, 353, 357, 361, 365]
# index_list_val = [350, 352, 354, 356, 358, 360, 362, 364, 366, 351, 355, 359, 363, 353, 357, 361, 365]
# index_list_test = [350, 352, 354, 356, 358, 360, 362, 364, 366, 351, 355, 359, 363, 353, 357, 361, 365]

# scene_name = '1d003b07bd'
# index_list_train = [4548, 4549, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564]
# index_list_val = [4548, 4549, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564]
# index_list_test = [4548, 4549, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564]

# scene_name = '5f99900f09'
# index_list_train = [i for i in range(5430, 5461)]
# index_list_val = [i for i in range(5430, 5461)]
# index_list_test = [i for i in range(5430, 5461)]

# scene_name = '49a82360aa'
# index_list_train = [131, 133, 135, 137, 139, 141, 143, 145, 132, 136, 140, 144, 134, 138, 139, 142]
# index_list_val = [131, 133, 135, 137, 139, 141, 143, 145, 132, 136, 140, 144, 134, 138, 139, 142]
# index_list_test = [131, 133, 135, 137, 139, 141, 143, 145, 132, 136, 140, 144, 134, 138, 139, 142]

scene_name = '355e5e32db'
index_list_train = [8723, 8725, 8727, 8729, 8731, 8733, 8735, 8737, 8739, 8741, 8743, 8745, 8747, 8749, 8751, 8753, 8755, 8757, 8759, 8761, 8763, 8765, 8767, 8769]
index_list_val = [8723, 8725, 8727, 8729, 8731, 8733, 8735, 8737, 8739, 8741, 8743, 8745, 8747, 8749, 8751, 8753, 8755, 8757, 8759, 8761, 8763, 8765, 8767, 8769]
index_list_test = [8723, 8725, 8727, 8729, 8731, 8733, 8735, 8737, 8739, 8741, 8743, 8745, 8747, 8749, 8751, 8753, 8755, 8757, 8759, 8761, 8763, 8765, 8767, 8769]

# scene_name = '6b40d1a939'
# start = 4982
# end = 5001

# scene_name = '6ee2fc1070_0'
# start = 7872
# end = 7890

# scene_name = '6ee2fc1070_2'
# start = 8445
# end = 8478

# scene_name = '6ee2fc1070_3'
# start = 8590
# end = 8620

# scene_name = '6f1848d1e3'
# start = 1373
# end = 1395

# new ones
# scene 6f12492455
# scene_name = '6f12492455_0'
# start = 7500
# end = 7525
#
# scene_name = '6f12492455_1'
# start = 7702
# end = 7724
#
# scene_name = '6f12492455_2'
# start = 7825
# end = 7845
#
# scene_name = '6f12492455_3'
# start = 8000
# end = 8025
#
# scene_name = '6f12492455_4'
# start = 8088
# end = 8110
#
# scene_name = '6f12492455_5'
# start = 8170
# end = 8200

# scene 7cd2ac43b4
# scene_name = '7cd2ac43b4_0'
# start = 3553
# end = 3588

# scene_name = '7cd2ac43b4_1'
# start = 3705
# end = 3724

# scene_name = '7cd2ac43b4_2'
# start = 3870
# end = 3890

# scene 7f4d173c9c
# scene_name = '7f4d173c9c_0'
# start = 1
# end = 25

# scene_name = '7f4d173c9c_1'
# start = 65
# end = 85

# scene_name = '7f4d173c9c_2'
# start = 450
# end = 480

# scene_name = '7f4d173c9c_3'
# start = 530
# end = 560

# scene_name = '7f4d173c9c_4'
# start = 850
# end = 880

# scene_name = '7f4d173c9c_5'
# start = 1435
# end = 1455

# scene_name = '7f4d173c9c_6'
# start = 1925
# end = 1960

# scene 8f82c394d6
# scene_name = '8f82c394d6_0'
# start = 7643
# end = 7666

# scene_name = '8f82c394d6_1'
# start = 7705
# end = 7725

# scene
# scene_name = '09c1414f1b_0'
# start = 4775
# end = 4810

# scene_name = '09c1414f1b_1'
# start = 4860
# end = 4890

# scene_name = '09c1414f1b_2'
# start = 5160
# end = 5190

# scene_name = '09c1414f1b_3'
# start = 5275
# end = 5300

# scene_name = '09c1414f1b_4'
# start = 5795
# end = 5820

# scene_name = '09c1414f1b_5'
# start = 6238
# end = 6265

#scene 25f3b7a318
# scene_name = '25f3b7a318_0'
# start = 6845
# end = 6869

# scene_name = '25f3b7a318_1'
# start = 6920
# end = 6940

# scene 54b6127146
# scene_name = '54b6127146_0'
# start = 7917
# end = 7940

# scene_name = '54b6127146_1'
# start = 7950
# end = 7970

# scene_name = '54b6127146_2'
# start = 8085
# end = 8105

# scene_name = '54b6127146_3'
# start = 8455
# end = 8480

# scene_name = '54b6127146_4'
# start = 8610
# end = 8630

# scene_name = '54b6127146_5'
# start = 8631
# end = 8655

# scene_name = '54b6127146_6'
# start = 8745
# end = 8770

# scene_name = '54b6127146_7'
# start = 8790
# end = 8815

# scene 95d525fbfd
# scene_name = '95d525fbfd_0'
# start = 6875
# end = 6900

# scene_name = '95d525fbfd_1'
# start = 7025
# end = 7050

# scene_name = '95d525fbfd_2'
# start = 7290
# end = 7320

# index lists
# index_list_train = [i for i in range(start, end)]
# index_list_val = [i for i in range(start, end)]
# index_list_test = [i for i in range(start, end)]

################### another set of focal length ###################
# scene_name = '0a7cc12c0e'
# index_list_train = [5842, 5844, 5846, 5848, 5850, 5851, 5883, 5885, 5887, 5890, 5843, 5849, 5884, 5888, 5845, 5847, 5886]
# index_list_val = [5842, 5844, 5846, 5848, 5850, 5851, 5883, 5885, 5887, 5890, 5843, 5849, 5884, 5888, 5845, 5847, 5886]
# index_list_test = [5842, 5844, 5846, 5848, 5850, 5851, 5883, 5885, 5887, 5890, 5843, 5849, 5884, 5888, 5845, 5847, 5886]

# scene_name = '0a184cf634'
# index_list_train = [6222, 6224, 6226, 6228, 6230, 6232, 6234, 6235, 6236, 6238, 6240, 6223, 6229, 6233, 6239, 6225, 6227]
# index_list_val = [6222, 6224, 6226, 6228, 6230, 6232, 6234, 6235, 6236, 6238, 6240, 6223, 6229, 6233, 6239, 6225, 6227]
# index_list_test = [6222, 6224, 6226, 6228, 6230, 6232, 6234, 6235, 6236, 6238, 6240, 6223, 6229, 6233, 6239, 6225, 6227]

# scene_name = '1f7cbbdde1'
# index_list_train = [i for i in range(5011, 5029)]
# index_list_val = [i for i in range(5011, 5029)]
# index_list_test = [i for i in range(5011, 5029)]

# scene_name = '3f1e1610de'
# index_list_train = [i for i in range(5943, 5964)]
# index_list_val = [i for i in range(5943, 5964)]
# index_list_test = [i for i in range(5943, 5964)]
################### another set of focal length ###################




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

        destination_folder = os.path.join(base_path, 'srn_cars', f'cars_{folder_type}', f'{scene_name}', folder_name)
        delete_files_in_folder(destination_folder)
    move_and_overwrite_files(scene_name, index_list, folder_type)
