import os
import open3d as o3d

# Define the path to your folder containing the RGBD images
folder_path = "/Users/yining/Desktop/rgbd-scenes/sub_table"  # Replace with the path to the folder containing your images

# List all files in the directory and sort them
all_files = sorted(os.listdir(folder_path))

# Filter out RGB and depth files into separate lists
rgb_files = [file for file in all_files if file.endswith('.png') and not file.endswith('_depth.png')]
depth_files = [file for file in all_files if file.endswith('_depth.png')]

# Ensure that there is a corresponding depth file for each RGB file
assert len(rgb_files) == len(depth_files), "The number of RGB and depth files do not match."

# Define your camera intrinsics here
f_x = 525  # Focal length in pixels
f_y = 525  # Focal length in pixels
c_x = 320  # Principal point x-coordinate in pixels
c_y = 240  # Principal point y-coordinate in pixels
width = 640  # Width of the image in pixels
height = 480  # Height of the image in pixels

# Creating the PinholeCameraIntrinsic object with the default parameters
intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, f_x, f_y, c_x, c_y)

point_clouds = []

# Iterate over paired RGB and depth images
for rgb_file, depth_file in zip(rgb_files, depth_files):
    # Create full paths to the files
    rgb_path = os.path.join(folder_path, rgb_file)
    depth_path = os.path.join(folder_path, depth_file)

    # Load the RGB and depth images
    rgb_image = o3d.io.read_image(rgb_path)
    depth_image = o3d.io.read_image(depth_path)

    # Create an RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image,
        depth_image,
        depth_scale=1000.0,  # Adjust if necessary
        depth_trunc=3.0,     # Adjust if necessary
        convert_rgb_to_intensity=False
    )

    # Create a point cloud from the RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    # Store the point cloud for further processing
    point_clouds.append(pcd)

# At this point, you can add further processing like registration and fusion.

# # Example: visualize all point clouds together (for a small number of point clouds)
# o3d.visualization.draw_geometries(point_clouds)

# Example: save all point clouds to files
for i, pcd in enumerate(point_clouds):
    o3d.io.write_point_cloud(f"output_point_cloud_{i}.ply", pcd)
    o3d.visualization.draw_geometries([pcd])
