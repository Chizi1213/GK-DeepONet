# GK-DeepONet
Gaussian Kernel DeepONet (GK-DeepONet) is a data-driven surrogate model for real-time prediction of structural deformation under dynamic loading and geometric variations. It overcomes limitations of standard DeepONet by introducing a learnable Gaussian kernel for nonlinear coupling between load-geometry inputs and spatial coordinates. Combined with the equivalent static load (ESL) method, GK-DeepONet efficiently handles transient dynamics without recurrent networks. Tested on 3D structures with up to 139,372 nodes, it achieves R² > 0.98 while accelerating inference from 390s to 0.26s per sample—enabling three orders of magnitude speedup and strong generalization across unseen geometries and loading conditions.  

Project structure:  
.
├── 1.Data_preprocessing/          # Data preparation pipeline  
│   ├── point_cloud.py             # Extract point cloud coordinates from simulation  
│   ├── step_times.py              # Extract time steps per experiment  
│   ├── displacement.py            # Extract true displacement fields  
│   ├── geom_time_load.py          # Generate geometry-time-load vectors (branch input)  
│   ├── time_coord_dis.py          # Generate coordinate-displacement pairs (trunk input & target)  
│   └── K-means_cluster.py         # Perform K-means clustering for node selection  
│  
├── 2.Geom-DeepONet/               # Standard DeepONet implementation  
│   ├── train_Geom_deeponet.py     # Train model  
│   └── predict_Geom_deeponet.py   # Run inference  
│  
└── 3.GK-DeepONet/                 # Proposed Gaussian Kernel DeepONet  
    ├── train_GK_deeponet.py       # Train model  
    └── predict_GK_deeponet.py     # Run inference  
