import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random
import copy
import os
import sys
import time


def get_intrinsic():
    f_x = 525  # Focal length in pixels
    f_y = 525  # Focal length in pixels
    c_x = 320  # Principal point x-coordinate in pixels
    c_y = 240
    K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]], dtype=np.float64)
    # Creating the PinholeCameraIntrinsic object with the default parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = K
    return intrinsic, K


def extract_last_number(file_name):
    # The last number is after the last underscore and before the extension
    parts = file_name.split('_')
    # The part we are interested in is the second last one, right before the extension
    number_part = parts[-2] if parts[-1].startswith('depth') else parts[-1]
    return int(number_part.split('.')[0])


def source_rbg(folder_path):
    source_rgb = []

    all_files = sorted(os.listdir(folder_path))
    rgb_files = [file for file in all_files if file.endswith('.png') and not file.endswith('_depth.png')]
    rgb_files = sorted(rgb_files, key=extract_last_number)
    depth_files = [file for file in all_files if file.endswith('_depth.png')]
    depth_files = sorted(depth_files, key=extract_last_number)

    assert len(rgb_files) == len(depth_files), "The number of RGB and depth files do not match."

    for rgb_file in rgb_files:
        rgb_path = os.path.join(folder_path, rgb_file)

        rgb_image = cv2.imread(rgb_path)
        source_rgb.append(rgb_image)

    return source_rgb


def point_cloud_list(folder_path):
    intrinsics, K = get_intrinsic()

    all_files = sorted(os.listdir(folder_path))
    rgb_files = [file for file in all_files if file.endswith('.png') and not file.endswith('_depth.png')]
    rgb_files = sorted(rgb_files, key=extract_last_number)
    depth_files = [file for file in all_files if file.endswith('_depth.png')]
    depth_files = sorted(depth_files, key=extract_last_number)

    assert len(rgb_files) == len(depth_files), "The number of RGB and depth files do not match."

    point_clouds = []
    rgb_images = []
    depth_images = []

    # Iterate over paired RGB and depth images
    for rgb_file, depth_file in zip(rgb_files, depth_files):
        rgb_path = os.path.join(folder_path, rgb_file)
        depth_path = os.path.join(folder_path, depth_file)

        rgb_image = o3d.io.read_image(rgb_path)
        depth_image = o3d.io.read_image(depth_path)
        rgb_images.append(rgb_image)
        depth_images.append(depth_image)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image,
            depth_image,
            depth_scale=999.99,  # Adjust if necessary
            depth_trunc=1.0,  # Adjust if necessary
            convert_rgb_to_intensity=False
        )

        # Create a point cloud from the RGBD image
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        point_clouds.append(point_cloud)

    return point_clouds, rgb_images, depth_images


def feature_extraction(folder_path):
    source_rgb = source_rbg(folder_path)
    kps = []
    descs = []
    for rgb_image in source_rgb:
        sift = cv2.SIFT_create(nOctaveLayers=20)
        kp, desc = sift.detectAndCompute(rgb_image, None)
        kps.append(kp)
        descs.append(desc)
    return kps, descs


# BFMatcher with default params
def bf_matcher(folder_path):
    kps, descs = feature_extraction(folder_path)

    bf = cv2.BFMatcher()
    all_good_matches = []

    for i in range(len(descs)):
        for j in range(i + 1, len(descs)):
            # Use knnMatch to find the top 2 matches for each descriptor
            matches = bf.knnMatch(np.asarray(descs[j], np.float32),
                                  np.asarray(descs[i], np.float32), k=2)
            good = []
            # Apply the ratio test to find good matches
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            # Add the good matches for this pair to the overall list
            all_good_matches.append(((j, i), good))
            # print(len(good))

    return all_good_matches


def pose_estimation(folder_path):
    kps, descs = feature_extraction(folder_path)

    intrinsics, K = get_intrinsic()

    point_clouds, rgb_images, depth_images = point_cloud_list(folder_path)
    all_good_matches = bf_matcher(folder_path)

    depth_scaling_factor = 999.99

    merged_pcd = copy.deepcopy(point_clouds[0])
    T_list = []
    for i in range(len(point_clouds) - 1):
        pt3d_1 = []
        pt3d_2 = []
        corr_list = []

        for count in range(len(all_good_matches)):
            if (all_good_matches[count][0] == (i + 1, i)):
                break

        if len(all_good_matches[count][1]) < 100:
            continue

        for j in range(len(all_good_matches[count][1])):
            kp1 = kps[i]
            kp2 = kps[i + 1]
            u2, v2 = kp2[all_good_matches[count][1][j][0].queryIdx].pt
            z2 = np.asarray(depth_images[i + 1], dtype=np.float64)[np.int32(v2)][np.int32(u2)] / depth_scaling_factor
            x2 = (u2 - K[0, 2]) * z2 / K[0, 0]
            y2 = (v2 - K[1, 2]) * z2 / K[1, 1]
            u1, v1 = kp1[all_good_matches[count][1][j][0].trainIdx].pt
            z1 = np.asarray(depth_images[i], dtype=np.float64)[np.int32(v1)][np.int32(u1)] / depth_scaling_factor
            x1 = (u1 - K[0, 2]) * z1 / K[0, 0]
            y1 = (v1 - K[1, 2]) * z1 / K[1, 1]
            pt3d_1.append([x1, y1, z1])
            pt3d_2.append([x2, y2, z2])
            corr_list.append([j, j])

        pc_1 = o3d.geometry.PointCloud()
        pc_2 = o3d.geometry.PointCloud()
        pc_1.points = o3d.utility.Vector3dVector(pt3d_1)
        pc_2.points = o3d.utility.Vector3dVector(pt3d_2)
        corres = o3d.utility.Vector2iVector(corr_list)

        icp_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pc_2, pc_1, corres, 0.5,
                                                                                            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                                                                                                False),
                                                                                            3, [],
                                                                                            o3d.pipelines.registration.RANSACConvergenceCriteria(
                                                                                                1000000, 0.999))
        T = icp_result.transformation
        T_list.append(T)
        npcd = copy.deepcopy(point_clouds[i+1])
        for j, item in enumerate(T_list):
            length = len(T_list)
            npcd = npcd.transform(T_list[length - j - 1])
        # Merge the two point clouds
        merged_pcd = merged_pcd + npcd

    # Downsample the merged point cloud to reduce density and remove duplicate points
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.003)

    last_part = os.path.basename(folder_path)
    o3d.io.write_point_cloud(f"/Users/yining/Sem1/MP/pythonProject/point_cloud/point_{last_part}.ply", merged_pcd)

    print(f"Finish Processing merged ply")

    return merged_pcd

def point_cloud_to_mesh_poisson(folder_path):
    merged_pcd = pose_estimation(folder_path)
    # file_path = f"/Users/yining/Sem1/MP/pythonProject/point_cloud/point_{last_part}.ply"
    # merged_pcd = o3d.io.read_point_cloud(file_path)
    
    merged_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
    # Apply Poisson Surface Reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(merged_pcd, depth=6)

    last_part = os.path.basename(folder_path)
    o3d.io.write_triangle_mesh(f"/Users/yining/Sem1/MP/pythonProject/mesh/mesh_poisson_{last_part}.obj", mesh)
    print(f"Finish Processing mesh")


def point_cloud_to_ball_pivoting(folder_path, radii=[0.0025, 0.005, 0.01, 0.02]):
    # Estimate normals
    merged_pcd = pose_estimation(folder_path)
    # file_path = f"/Users/yining/Sem1/MP/pythonProject/point_cloud/point_{last_part}.ply"
    # merged_pcd = o3d.io.read_point_cloud(file_path)

    merged_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        merged_pcd, o3d.utility.DoubleVector(radii))

    last_part = os.path.basename(folder_path)
    o3d.io.write_triangle_mesh(f"/Users/yining/Sem1/MP/pythonProject/mesh/mesh_ball_{last_part}.obj", mesh)
    print(f"Finish Processing mesh")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        input_file_path = sys.argv[1]
        print(f"Processing file: {input_file_path}")
        start_time = time.time()

        # merged_pcd = pose_estimation(input_file_path)
        point_cloud_to_ball_pivoting(input_file_path, radii=[0.0025, 0.005, 0.01, 0.02])
        point_cloud_to_mesh_poisson(input_file_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to run the code: {elapsed_time} seconds")

    else:
        print("No file path provided.")
