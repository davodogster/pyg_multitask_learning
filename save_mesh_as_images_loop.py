# conda activate pdal

import open3d as o3d
import copy
import numpy as np
import glob

meshs = glob.glob("D:\\Blender_random\\Sam\\june22_pre_cleaned\\*.stl")
print(meshs)

for mesh_ in meshs:
    mesh = o3d.io.read_triangle_mesh(mesh_) # works better!
    # mesh = o3d.io.read_triangle_mesh("D:\\Blender_random\\Tancred_Misc\\G10modl__branch_weight_2.0_bake_bend_0.2_age_25.stl") # works better!
    mesh_r = copy.deepcopy(mesh)
    R = mesh.get_rotation_matrix_from_xyz((np.pi / -2, 0, np.pi / -2)) # works well ? (np.pi / -2, 0, np.pi / -2))
    mesh_r.rotate(R, center=(0, 0, 0))
    # o3d.visualization.draw_geometries([mesh_r]) # [mesh, mesh_r]
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh_r)
    vis.update_geometry(mesh_r)
    vis.poll_events()
    vis.update_renderer()
    new_name = mesh_.replace(".stl", ".PNG")
    vis.capture_screen_image(new_name)
    vis.destroy_window()

# o3d.visualization.draw_geometries([mesh])
# o3d.visualization.ViewControl.change_field_of_view()