# 3D_reconstruction
This project is a PyTorch implementation of [Neural RGB-D Surface Reconstruction](https://dazinovic.github.io/neural-rgbd-surface-reconstruction/static/pdf/neural_rgbd_surface_reconstruction.pdf), which is a novel approach for 3D reconstruction that combines implicit surface representations with neural radiance fields

# Dataset
https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes

# Point cloud
Use open3d to generate a point cloud from a RGBD image.

And then use SIFT, ICP to register the point clouds.

# Mesh
Use 2 methods to generate a mesh from a point cloud: Poisson and Ball pivoting.

