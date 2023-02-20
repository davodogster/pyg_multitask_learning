#user skeletor to derive skeletons for my trees
import skeletor as sk
import sys
import trimesh
from os.path import exists
#import bpy

#get the filenames to use

for file in sys.argv[1::]:
    print("doing file '"+file+"'")
    basename=file
    if file.endswith(".stl"):
        basename=file[:-4]
    print("basename:"+basename)
    if exists(basename+".swc"):
        print("skipping, output exists")
        continue
    #if the file did NOT exist, do the rest
    mesh=trimesh.load(file)
    #mesh=bpy.ops.import_mesh.stl(filepath=file)
    fixed=sk.pre.fix_mesh(mesh, remove_disconnected=5,inplace=False)
    cont = sk.pre.contract(fixed, epsilon=0.1,SL=10,WH0=0.01,operator="cotangent")
    skel=sk.skeletonize.by_teasar(cont,inv_dist=1)
    skel.mesh = fixed #reset to un-contracted mesh
    sk.post.radii(skel, method='knn')
    skel.save_swc(basename+".swc")
    #skel.show(mesh=True) #de-comment to see the skeletons
    
    
