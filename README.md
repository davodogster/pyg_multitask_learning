# pyg_multitask_learning

I mainly just use github to backup my code.


Install pytorch cuda that works with your conda environment. Also install pytorch_geometric.
Then Install trimesh and skeletor packages.

Point cloud Regression script takes in directories of folders with XYZ files and reads them into a point cloud PyG Dataset.


Skeletor Skeleton Regression script takes in directories of .stl mesh files and reads them into a point cloud PyG Dataset. Where skeletor edges = pyg edge_index and skeletor vertices = pyg.pos.
