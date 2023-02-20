#!/usr/bin/env python
# coding: utf-8



import glob
import math
import os
import numpy as np
import skeletor as sk
import trimesh


# os.chdir("/treesim")

# path to blend and stl files (individual trees)
dataset_path = '/treesim/simulation_Feb_2023/Output_X0_Y0/'
unique_params = os.listdir(dataset_path)
print(unique_params[0:3])
len(unique_params)

# os.mkdir('/treesim/simulation_Feb_2023/Output_X0_Y0/Grove10x10x10_skel_npy/')

# mesh_path = os.path.join(dataset_path + unique_params[i])
# mesh = trimesh.load(mesh_path)

data_list = []
for i in range(0, len(unique_params)):
    if ".stl" in unique_params[i]:
        print("index", i)
        print(unique_params[i])
        data = []
        mesh_path = os.path.join(dataset_path + unique_params[i])
        mesh = trimesh.load(mesh_path)
        fixed=sk.pre.fix_mesh(mesh, remove_disconnected=5,inplace=False)
        cont = sk.pre.contract(fixed, epsilon=0.1,SL=10,WH0=0.01,operator="cotangent") # param values from Tancred
        skel=sk.skeletonize.by_teasar(cont,inv_dist=1)
        skel.mesh = fixed #reset to un-contracted mesh
        # save as npy arrays
        save_path = mesh_path.replace(".stl", "_vert.npy")
        save_path = save_path.replace(dataset_path, # files_dir
                                     "/treesim/simulation_Feb_2023/Output_X0_Y0/Grove10x10x10_skel_npy/") # output_dir 
        np.save(save_path, skel.vertices)
        save_path = mesh_path.replace(".stl", "_edges.npy").replace(dataset_path, #  # files_dir
                                     "/treesim/simulation_Feb_2023/Output_X0_Y0/Grove10x10x10_skel_npy/")
        np.save(save_path, skel.edges)
    else:
        print("skip")  


print("DONE")

