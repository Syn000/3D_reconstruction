# 3D_reconstruction
This project is a PyTorch implementation of 3D RECONSTRUCTION CODING CHALLENGE.

# Dataset
https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes

# Point cloud
For each RGBD image, use open3d to generate a point cloud.

And then use SIFT, BFMatcher, ICP to register the generated point clouds, and then create one point cloud.

The generated point clouds for 8 scenes can be downloaded via the following link:.

# Mesh
Use 2 methods to generate a mesh from a point cloud: Poisson and Ball pivoting.

The meshs can be downloaded via the following link:.

# Textured mesh
Use Blender to create and project a UV image to apply texture to the mesh.

The texture image is downloaded via:.

The textured mesh can be downloaded via the following link:.
