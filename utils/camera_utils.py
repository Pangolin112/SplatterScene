import numpy as np

def get_loop_cameras(num_imgs_in_loop, radius=2.0, 
                     max_elevation=np.pi/6, elevation_freq=0.5,
                     azimuth_freq=2.0):

    all_cameras_c2w_cmo = []

    for i in range(num_imgs_in_loop):
        azimuth_angle = np.pi * 2 * azimuth_freq * i / num_imgs_in_loop
        elevation_angle = max_elevation * np.sin(
            np.pi * i * 2 * elevation_freq / num_imgs_in_loop)
        x = np.cos(azimuth_angle) * radius * np.cos(elevation_angle)
        y = np.sin(azimuth_angle) * radius * np.cos(elevation_angle)
        z = np.sin(elevation_angle) * radius

        camera_T_c2w = np.array([x, y, z], dtype=np.float32)

        # in COLMAP / OpenCV convention: z away from camera, y down, x right
        camera_z = - camera_T_c2w / radius
        up = np.array([0, 0, -1], dtype=np.float32)
        camera_x = np.cross(up, camera_z)
        camera_x = camera_x / np.linalg.norm(camera_x)
        camera_y = np.cross(camera_z, camera_x)

        camera_c2w_cmo = np.hstack([camera_x[:, None], 
                                    camera_y[:, None], 
                                    camera_z[:, None], 
                                    camera_T_c2w[:, None]])
        camera_c2w_cmo = np.vstack([camera_c2w_cmo, np.array([0, 0, 0, 1], dtype=np.float32)[None, :]])

        all_cameras_c2w_cmo.append(camera_c2w_cmo)

    return all_cameras_c2w_cmo