# 3D_reconstruction
This project is a PyTorch implementation of 3D RECONSTRUCTION CODING CHALLENGE.

# Dataset
https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes

Due to constraints on time and CPU processing power, I chose to use a subset of the available dataset for my experiments. Each subset consists of sequences of 45 RGB images and their matching depth images from each scene.

# Run 
```python "rgbd_mesh(multi_view).py" <PATH_TO_YOUR_DATA>```

# Point cloud
For each RGBD image, use open3d to generate a point cloud. And then use SIFT, BFMatcher, ICP to register the generated point clouds, and then merge the registerd point clouds to create one point cloud.

The generated point clouds for 8 scenes can be downloaded via the following link: https://drive.google.com/file/d/1s8dhvDcCzLFQO6RoA68QyPXaIJee888v/view?usp=sharing.

# Mesh
Provide 2 methods to generate a mesh from a point cloud: Poisson and Ball pivoting.

The meshs can be downloaded via the following link: https://drive.google.com/file/d/1gbo2Ps-L7hlG7IH8emO-dlMxAOqai2nm/view?usp=sharing.

![image](https://github.com/Syn000/3D_reconstruction/blob/main/Pros_Cons.png)

# Textured mesh
Used Blender to create and project a UV image to apply texture to the mesh.

The texture image is downloaded via: https://ambientcg.com/.

The generated textured mesh sample can be downloaded via the following link: https://drive.google.com/file/d/1uIOqrruxm1cfpXJ5rEbj9gxrruUwu3wT/view?usp=sharing.

# Reference
https://arxiv.org/pdf/2001.05119.pdf

https://github.com/PHANTOM0122/3D_Object_Reconstruction/tree/main?tab=readme-ov-file

https://github.com/dazinovic/neural-rgbd-surface-reconstruction/tree/main
