# pyg_multitask_learning

Point Cloud Multitask Regression using GNN in Pytorch Geometric.

Install pytorch cuda that works with your conda environment. Also install pytorch_geometric.
Then onstall trimesh and skeletor packages.

Point cloud Regression script takes in directories of folders with XYZ files and reads them into a point cloud PyG Dataset. It then trains a multitask regression network to estimate multiple continuos tree parameters. PCDs_newestdata_training_v0 (4).ipynb has end to end code for the pipeline to train a 3-task model. 

Example for 2-task learning:

![image](https://user-images.githubusercontent.com/46079516/177441575-77a14c5c-9eef-46b9-857c-fc6727bc29e3.png)
![image](https://user-images.githubusercontent.com/46079516/177441609-d9089f68-2ea5-41ad-b720-2b06fa0bcb09.png)

![image](https://user-images.githubusercontent.com/46079516/177441693-dd1a6a5b-df69-4c20-9775-0250bb10fff5.png)

![image](https://user-images.githubusercontent.com/46079516/177441654-3675a820-445a-4883-bdf2-0758ee386b1c.png)

Skeletor Skeleton Regression script takes in directories of .stl mesh files and reads them into a point cloud PyG Dataset. Where skeletor edges = pyg edge_index and skeletor vertices = pyg.pos.

![image](https://user-images.githubusercontent.com/46079516/177441522-cfd246c1-bfb7-4041-9d55-6d64a0705fe9.png)

save_mesh_as_images_loop.py will iterate over mesh files and take a side view images of trees and save them as image files.

mesh_to_skeletor_numpy will convert mesh files into skeletor format as numpy arrays which can be used to train a GNN on skeletons instead of PCDs. This allows for faster training as skeletons are less complex and smaller files than PCDs.
