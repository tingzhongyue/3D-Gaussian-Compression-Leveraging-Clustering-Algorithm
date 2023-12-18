# 3D-Gaussian-Compression-Leveraging-Clustering-Algorithm

As a representation method for displaying 3D scenes, 3D Gaussian allows optimization with state-of-the-art (SOTA) visual quality and competitive training times. In comparison to previous explicit scene representations, although the anisotropic Gaussians are capable of modelling complex shapes with a lower number of parameters, it still requires a large amount of memory and storage.

This project is a faithful PyTorch implementation of 3DGS that reproduces the results while the storage volume is reduced to 0.24 times the original size. The code is based on authors' implementation [here](https://github.com/graphdeco-inria/gaussian-splatting), and has been tested to match it numerically.

## Overview

 The codebase has 3 main components: 

- Original code implementation
- Clustering 3DGS points using K-Means
- Clustering 3DGS points using MeanShift

## How to Run?

### train a vanilla Gaussian

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to store the trained model>--eval
```

 After training for 30k iterations (~20 minutes on a single 4090), you can find the following pointcloud at 

```/data/output/point_cloud/iteration_30000/point_cloud.ply```

This position depends on the path to store the trained model. The path used here is 

```shell
python train.py -s data -m data/output --eval
```



### Clustering 3DGS points using K-Means

After getting the vanilla Gaussian points, you need to run ```kmeans.py```

```shell
python kmeans.py --file <<path to store Gaussian point cloud> --eval <whether to test>
```

Then, in the same path, you can find the following pointcloud at 

```/data/output/point_cloud/iteration_30000/point_cloud.ply```

Finally, if you need to render the image, you need to run ```render_vq.py```

```shell
python render.py -m data/output --eval
```

### Clustering 3DGS points using K-Means

The operation method here is the same as the above method.
