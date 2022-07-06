#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch_geometric
import torch
import matplotlib.pyplot as plt
import scipy
import random
from random import Random
import skeletor
import trimesh
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import math
import random
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import open3d as o3d
torch.cuda.empty_cache()


# In[48]:


# import glob
# dataset_path = '/home/scion.local/davidsos/Documents/Grove_Regression/Tancred_Trees_Mesh/'
# unique_params = os.listdir(dataset_path)
# # print(unique_params)
# # meshs = glob.glob(dataset_path + "/*.stl")

# data_list = []

# print(unique_params[0])
# data = []

# mesh_path = os.path.join(dataset_path + unique_params[i])
# mesh = trimesh.load(mesh_path)
# mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
# fixed = skeletor.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
# skel = skeletor.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
# print(skel.vertices)
# print(skel.edges)
# pos = torch.from_numpy(skel.vertices).to(torch.float)
# edge_index = torch.from_numpy(skel.edges.T).to(torch.float)
# print(edge_index)


# In[1]:


# for i in range(0, len(unique_params)):
#     name = unique_params[i] # .split("/")[7]
#     print(name)
#     ag = name.split("-")[2].replace("_age_Conifer_", "").replace("_ageConifer", "").replace("_age_Conifer_", "").replace("_agConifer ", "").replace("_age_Conifer", "").replace("_agConifer_", "").replace("_ageConifer_", "")
#     ag = ag.strip(" ")
#     if "0-0_ag" in name:
#         ag = '0'
#     elif ag == '0':
#         ag = '100'
#     ag = ag.ljust(2, '0')
#     print(ag)
# #     data.ag = torch.from_numpy(np.array(int(ag))).to(torch.float)


# In[59]:


# for i in range(0, len(unique_params)):
#     data = []

#     name = unique_params[i] # .split("/")[7]
#     print(name)
#     branch = name.split("-")[1].replace("_branch_chance_only_terminal_0", "").replace("_branch_chance_only_terminal_1", "")
#     print(branch)
#     branch = branch.strip(" ")
#     if "0-0_branch" in name:
#         branch = '0'
#     elif branch == '0':
#         branch = '100'
#     branch = branch.ljust(2, '0')
#     print(branch)


# In[2]:


import glob
dataset_path = '/home/scion.local/davidsos/Documents/Grove_Regression/Tancred_Trees_Mesh/'
unique_params = os.listdir(dataset_path)
print(len(unique_params))


# In[3]:



# import glob
# dataset_path = '/home/scion.local/davidsos/Documents/Grove_Regression/Tancred_Trees_Mesh/'
# unique_params = os.listdir(dataset_path)
# # print(unique_params)
# # meshs = glob.glob(dataset_path + "/*.stl")

# data_list = []

# for i in range(0, len(unique_params)):
#     print("index", i)
#     print(unique_params[i])
#     data = []

#     mesh_path = os.path.join(dataset_path + unique_params[i])
#     mesh = trimesh.load(mesh_path)
#     mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
#     fixed = skeletor.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
#     skel = skeletor.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
#     #print(skel.vertices)
#     #print(skel.edges)
#     pos = torch.from_numpy(skel.vertices).to(torch.float)
#     edge_index = torch.from_numpy(skel.edges.T).to(torch.int).type(torch.LongTensor)
    
#     name = unique_params[i] # .split("/")[7]
#     print(name)
#     branch = name.split("-")[1].replace("_branch_chance_only_terminal_0", "").replace("_branch_chance_only_terminal_1", "")
#     branch = branch.strip(" ")
#     if "0-0_branch" in name:
#         branch = '0'
#     elif branch == '0':
#         branch = '100'
#     branch = branch.ljust(2, '0')
#     ag = name.split("-")[2].replace("_age_Conifer_", "").replace("_ageConifer", "").replace("_age_Conifer_", "").replace("_agConifer ", "").replace("_age_Conifer", "").replace("_agConifer_", "").replace("_ageConifer_", "")
#     ag = ag.strip(" ")
#     if "0-0_ag" in name:
#         ag = '0'
#     elif ag == '0':
#         ag = '100'
#     ag = ag.ljust(2, '0')
#     ID = name.split(".")[1]
#     data = Data(pos=pos, edge_index = edge_index)
#     data.ID = ID
#     data.chance = torch.from_numpy(np.array(float(branch))).to(torch.float)
#     data.ag = torch.from_numpy(np.array(int(ag))).to(torch.float)
    
#     print(data)
#     data_list.append(data)
# print("DONE")


# # print(np.array(data).shape)
# # print(np.array(data))
# # print(data_list)

# # print(data.shape, data)
# # pcs = glob.glob(dataset_path + '/*.txt')


# In[ ]:





# In[4]:


len(data_list)


# In[5]:


data_list


# In[5]:


# data_list[0].ag


# In[6]:


# need to preserve distance/diam for each stem
# so obtain scaling values then indivdually apply to each point cloud


# In[46]:


data_list[0].edge_index


# In[96]:


data_list[0].edge_index


# In[52]:


# # Install required packages.
# # Helper functions for visualization.
# %matplotlib inline
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 1], pos[~mask, 2], s=20, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 1], pos[mask, 2], s=20, zorder=1000)
    plt.axis('off')
    plt.show()
    
    
# # from torch_geometric.transforms import FixedPoints
# # transform = FixedPoints(num=200)
# # transform(data_list)  # Explicitly transform data.
# import random 
# import torch_geometric.transforms as T

# # data = random.sample(data_list[0], 200)
# # transform = T.Compose([T.Center()]) # , T.NormalizeRotation(max_points=1000)])
# dataset = data_list.copy()
# # dataset = transform(dataset) 

# data = dataset[0]
# print(data)
# visualize_points(data.pos) # , data.edge_index)

# data = dataset[4]
# print(data)
# visualize_points(data.pos) # , data.edge_index)


# In[98]:


# from torch_cluster import knn_graph, knn

# dataset = data_list.copy()
# data = dataset[5]
# # data.edge_index2 = knn_graph(x=pos, k = 6)
# print(data.edge_index.shape)
# visualize_points(data.pos, edge_index=data.edge_index)

# # data = dataset[4]
# # print(data.edge_index.shape)
# # visualize_points(data.pos, edge_index=data.edge_index) 

# # edge index shape is n_points*K = 500*6 = 3000


# In[51]:


data.edge_index2.shape


# In[52]:


data.edge_index2.dtype


# In[6]:


data.edge_index.dtype


# In[36]:


# data = dataset[1]
# data.edge_index = knn_graph(data.pos, k=5)
# data


# In[12]:


# ax = plt.axes(projection='3d')
# fig = plt.figure(figsize = (14, 10))

# # Data for a three-dimensional line
# # zline = np.linspace(0, 15, 1000)
# # xline = np.sin(zline)
# # yline = np.cos(zline)
# # ax.plot3D(xline, yline, zline, 'gray')

# i = 278

# # Data for three-dimensional scattered points
# ax.scatter3D(dataset[i].pos[:, 0], dataset[i].pos[:, 1], dataset[i].pos[:, 2], c=dataset[i].pos[:, 2]) # , cmap='Greens');
# ax.view_init(40, 60)
# plt.show()


# In[67]:


# ax = plt.axes(projection='3d')
# fig = plt.figure(figsize = (14, 10))

# # Data for a three-dimensional line
# # zline = np.linspace(0, 15, 1000)
# # xline = np.sin(zline)
# # yline = np.cos(zline)
# # ax.plot3D(xline, yline, zline, 'gray')

# i = 20 # 202 is good
# data = data_list[i]

# # transform = T.Compose([T.NormalizeRotation(max_points=1000)]) # T.Center(),

# # transform = T.RandomRotate(degrees = [0, 180], axis = 2) #,
# # #                             T.RandomRotate(degrees = [-20, 20], axis = 0),
# # #                             T.RandomRotate(degrees = [-20, 20], axis = 1)])

# # data = transform(data_list[i])

# # data = dataset[i]
# # Data for three-dimensional scattered points
# ax.scatter3D(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2], c=data.pos[:, 2]) # , cmap='Greens');
# # ax.view_init(40, 60)
# ax.set_xlabel('$X$', fontsize=20, rotation=150)
# ax.set_ylabel('$Y$', fontsize=20, rotation=150)
# ax.set_zlabel('$Z$', fontsize=30, rotation=150)

# plt.show()


# In[68]:


print("YOOLO")


# In[34]:


# add more data to list

# train-val-test split BEFORE DATA AUGMENTATION STEP !!

dataset = data_list.copy()
seed = 239811888989 # 105
Random(seed).shuffle(dataset)
data_train = dataset[0:-60] # dataset[0:96]
data_val = dataset[-60:-30]
data_test = dataset[-30:]

y_orig = [data_train[i].chance.item() for i in range(0, len (data_train))] # new_train
# data_train[0].y.item()
y_orig_val = [data_val[i].chance.item() for i in range(0, len (data_val))]
y_orig_test = [data_test[i].chance.item() for i in range(0, len (data_test))]

# y_df = pd.DataFrame(y_orig) # , y_orig)

# y_df.describe()
# y_df.boxplot()
plt.boxplot([y_orig, y_orig_val, y_orig_test])


# In[25]:


# print(data_train)


# In[15]:


print(len(data_train),len(data_val), len(data_test))


# In[16]:


data_train


# In[23]:


# import os
# os.mkdir('/home/scion.local/davidsos/Documents/ForInstance_152_0_4m')


# In[3]:


import torch
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T


class DBHDataSet_train_aug(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Chicken']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = data_train

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
# transform = T.Compose([T.NormalizeRotation(max_points=1000)]) # T.Center(),
fly_transform = T.Compose([
    T.RandomRotate(degrees = [-180, 180], axis = 2),
#                         T.RandomRotate(degrees = [-10, 10], axis = 0),
#                         T.RandomRotate(degrees = [-10, 10], axis = 1)
])

DBH_DS = DBHDataSet_train_aug(root = "/home/scion.local/davidsos/Documents/Grove_Regression/train_chance_50000_KNN6_1", 
                              transform = fly_transform, pre_transform=T.KNNGraph(k=6)) # , , transform = fly_transform
 


# In[ ]:





# In[4]:


import torch
from torch_geometric.data import InMemoryDataset, download_url


class DBHDataSet_val(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Chicken']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = data_val

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# fly_transform = T.Compose([ #T.NormalizeRotation(max_points=500),
#     T.RandomRotate(degrees = [-60, 60], axis = 2), 
#                             T.RandomRotate(degrees = [-10, 10], axis = 0),
#                             T.RandomRotate(degrees = [-10, 10], axis = 1)])

DBH_DS_val = DBHDataSet_val(pre_transform=T.KNNGraph(k=6), root = "/home/scion.local/davidsos/Documents/Grove_Regression/val_chance_50000_KNN6_1") # , 
                   # pre_transform=transform) # , transform = fly_transform


# In[5]:


import torch
from torch_geometric.data import InMemoryDataset, download_url


class DBHDataSet_test(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Chicken']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = data_test

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
# transform = T.Compose([T.Center(), T.NormalizeRotation(max_points=1000)])

DBH_DS_test = DBHDataSet_test(pre_transform=T.KNNGraph(k=6), root = "/home/scion.local/davidsos/Documents/Grove_Regression/test_chance_50000_KNN6_1") #, 
                    # pre_transform=transform)


# In[ ]:





# In[7]:


len(DBH_DS)


# In[28]:


# transform = T.Compose([T.Center()])
# fly_transform = T.Compose([ #T.NormalizeRotation(max_points=500),
#     T.RandomRotate(degrees = [-180, 180], axis = 2), 
#                             T.RandomRotate(degrees = [-20, 20], axis = 0),
#                             T.RandomRotate(degrees = [-20, 20], axis = 1)])

# DBH_DS = DBHDataSet(root = "/home/scion.local/davidsos/Documents/DBH_Train_InMemory", 
#                     pre_transform=transform, transform = fly_transform)


# In[7]:


DBH_DS[0].pos # works on the fly!


# In[8]:


DBH_DS_val[0].pos


# In[31]:


DBH_DS[0].y


# In[32]:


DBH_DS[0].ID


# In[33]:


len(DBH_DS)


# In[34]:


len(DBH_DS_test)


# In[ ]:





# In[35]:


# ## sample DBH_DS

# DBH_DS_sample = random.sample(list(DBH_DS), 500)
# y_orig_sample = [new_train[i].y.item() for i in range(0, len (list(DBH_DS_sample)))]
# plt.boxplot([y_orig_sample, y_orig_val, y_orig_test])


# In[6]:


TRAIN_BS = 10
train_loader = DataLoader(DBH_DS, batch_size=TRAIN_BS, shuffle=True, num_workers=24) # BS must be equal to (or multiple?) of hidden_channels??
val_loader = DataLoader(DBH_DS_val, batch_size=24, shuffle = False, num_workers=24)
test_loader = DataLoader(DBH_DS_test, batch_size=24, shuffle = False, num_workers=25)
print("DATALOADERS READY")


# In[ ]:





# In[7]:


## Test RMSE if Guessing Training mean

len(y_orig)
import statistics

statistics.mean(y_orig), statistics.mean(y_orig_test)

y_test_guess = []
for i in range(0, len(y_orig_test)):
    y_test_guess.append(statistics.mean(y_orig))


## Guessing RMSE (because squared = False)

rms = mean_squared_error(y_orig_test, y_test_guess, squared=False)
print(rms) # chance 22.8


# In[8]:


# average number of points in test examples:
points = []

for data in DBH_DS_test:
    points.append(len(data.pos[:,0]))

print(statistics.mean(points))


# In[14]:


import torch
import torch.nn.functional as F
from torch_cluster import knn_graph, knn
from torch_geometric.nn import global_max_pool, global_mean_pool, global_sort_pool #, TopKPooling
from torch_cluster import fps

# idx = fps(positions, batch, ratio)
# row, col = knn(pos, pos[idx], n_neighbors, batch, batch[idx])
# out = scatter(x[row], col, dim=0, dim_size=idx.size(0), reduce='max')

from torch.nn import Sequential, Linear, ReLU #, Conv1d
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_isolated_nodes, dropout_adj

# K = 16 # no KNN graph cause we already have the edge_index :)
LOOP = True
FILTERS = 32
# scaler = "None"
TOPK = 0.5
Dropout = True

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max') # max
        
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + 3, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))
        
    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.
    

class PointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        self.conv1 = PointNetLayer(3, FILTERS) # 32 
        self.conv2 = PointNetLayer(FILTERS, FILTERS)# 32
        self.conv3 = PointNetLayer(FILTERS, FILTERS)# 32
#         self.dropout40 = torch.nn.Dropout(0.5)
#         self.dropout60 = torch.nn.Dropout(0.5)
        self.regressor1 = Linear(FILTERS, 1)  # 32, 1 # in_features, out_features
        self.regressor2 = Linear(FILTERS, 1)  # 32, 1 # in_features, out_features

#         self.pool = TopKPooling(FILTERS, 0.5)
        
    def forward(self, pos, edge_index, batch): # pos
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        # FPS from https://github.com/dragonbook/pointnet2-pytorch/blob/master/model/pointnet2_part_seg.py
#         idx = fps(pos, batch, ratio=0.7)  # self.sample_ratio
#         # Group(Build graph)
#         row, col = radius(pos, pos[idx], self.radius, batch, batch[idx], max_num_neighbors=self.max_num_neighbors)
#         edge_index = torch.stack([col, row], dim=0)
#         edge_index = knn_graph(pos, k=K, batch=batch, loop=LOOP, num_workers=24) # important set K here! Wang uses K=10
#         edge_index = remove_isolated_nodes(edge_index)[0]
#         edge_index = knn_graph(pos, k=K, batch=batch, loop=LOOP) # important set K here! Wang uses K=10
#         edge_index2 = remove_isolated_nodes(edge_index)[0]
        edge_index = remove_isolated_nodes(edge_index)[0]

        
#         edge_index = knn_graph(pos, k=K, batch=batch, loop=LOOP) # important set K here! Wang uses K=10
#         edge_index2 = remove_isolated_nodes(edge_index)[0]
        
#         edge_index = dropout_adj(edge_index = edge_index, p=0.2, training = self.training)[0] # added

        # 3. Start bipartite message passing.
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
#         h = F.dropout(h, p=0.8, training = self.training)

#         edge_index = dropout_adj(edge_index = edge_index, p=0.1, training=self.training)[0]
#         edge_index = remove_isolated_nodes(edge_index)[0]

#         idx = fps(pos, batch, 0.1)
#         row, col = knn(pos, pos[idx], k=K, batch, batch[idx1) # , loop=LOOP)
#         h = scatter(h[row], col, dim=0, dim_size=idx.size(0), reduce='max')
#         h = F.dropout(h, p=0.2)
#         h, edge_index, _, batch, _, _ = self.pool(h, edge_index,  batch)
        
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
#         h = self.dropout60(h)
#         h = F.dropout(h, p=0.8, training = self.training)

#         edge_index = dropout_adj(edge_index = edge_index, p=0.1)[0]
#         edge_index = remove_isolated_nodes(edge_index)[0]

        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
#         h = F.dropout(h, p=0.8, training = self.training)

        
        # 4. Global Pooling.
        h = global_mean_pool(h, batch)  # pools the features [num_examples, hidden_channels]
        
        # 5. Regressors.
        return [self.regressor1(h), self.regressor2(h)]


# model = PointNet()
# print(model)


# In[36]:


from torch_geometric.nn import GraphConv
import torch.nn.functional as F
# from torch_cluster import knn_graph, knn
from torch_geometric.nn import global_max_pool, global_mean_pool, global_sort_pool #, TopKPooling
from torch.nn import Sequential, Linear, ReLU #, Conv1d
# from torch_geometric.nn import MessagePassing
# from torch_geometric.transforms import BaseTransform
# from torch_geometric.utils import remove_isolated_nodes, dropout_adj


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(3, hidden_channels) # num_node_features = 3
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 1)
        self.lin2 = Linear(hidden_channels, 1)


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.1, training=self.training)
        x1 = self.lin1(x)
        x2 = self.lin2(x)
        
        return [x1, x2]


# In[37]:


# torch.cuda.empty_cache() 
device = torch.device('cuda') #  if torch.cuda.is_available() else 'cpu')
# model = PointNet().to(device)
K = 6 # no KNN graph cause we already have the edge_index :)
LOOP = True
FILTERS = 64
TOPK = 0.5
Dropout = True

model = GNN(hidden_channels=FILTERS).to(device)

LR = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# criterion = torch.nn.L1Loss(reduction='mean') # average the loss across the batch & Define loss criterion.
loss_chance = torch.nn.MSELoss(reduction='mean')
loss_ag = torch.nn.MSELoss(reduction='mean')


def train(model, optimizer, loader):
    model.train()
    total_loss = 0
    count = 0
    for data in loader:
        optimizer.zero_grad()  # Clear gradients.
        logits = model(data.pos.to(device), data.edge_index.to(device), data.batch.to(device))  # Forward pass. #data.pos
        loss1 = loss_chance(logits[0].to(torch.float32).squeeze(-1).to(device), data.chance.to(torch.float32).to(device)) # Loss computation. # logits[1]
        loss2 =  loss_ag(logits[1].to(torch.float32).squeeze(-1).to(device), data.ag.to(torch.float32).to(device)) # Loss computation. # logits[1]
        loss = loss1 + loss2
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
#         print(loss.item())
#         print(data.num_graphs)
        total_loss += loss.item() # * data.num_graphs (num_graphs is just batch size.. weird)
        count+=1

    return math.sqrt(total_loss / count) # len(train_loader.dataset)) math.sqrt(


## THE GUESSING RMSE for whole data (not train) 12.8
# @torch.no_grad()
# def test_hub(model, loader):
#     model.eval()
#     total_MSE = 0
#     count = 0
#     for data in loader:
#         preds = model(data.pos.to(device), data.edge_index.to(device), data.batch.to(device))
# #         y_output_pred.squeeze(-1)
#         loss = criterion_hub(preds.to(torch.float32).squeeze(-1).to(device), data.chance.to(torch.float32).to(device)) # Loss computation.
#         total_MSE += loss.item() #  * data.num_graphs
#         count+=1
        
#     return total_MSE / count ## math.sqrt(


@torch.no_grad()
def test_rmse(model, loader):
    model.eval()
    total_MSE = 0
    count = 0
    for data in loader:
        preds = model(data.pos.to(device), data.edge_index.to(device), data.batch.to(device))
#         y_output_pred.squeeze(-1)
        loss1 = loss_chance(preds[0].to(torch.float32).squeeze(-1).to(device), data.chance.to(torch.float32).to(device)) # Loss computation. # logits[1]
        loss2 =  loss_ag(preds[1].to(torch.float32).squeeze(-1).to(device), data.ag.to(torch.float32).to(device)) # Loss computation. # logits[1]
        loss = loss1 + loss2
        total_MSE += loss.item() #  * data.num_graphs
        count+=1
        
    return math.sqrt(total_MSE / count) ## math.sqrt(


print("DONE")


# In[38]:


print("K:", None, "LR:", LR, "Tr:", 96, "Test:", "24:", "Seed:", 105, "Filters:",
      FILTERS, "LOOP:", LOOP,  "BS:", TRAIN_BS, "Dropout:", Dropout)
import math

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 75, verbose = True)

best_val_mse = 1000
train_loss_hist = []
val_loss_hist = []
test_loss_hist = []

for epoch in range(1, 100000):
    loss = train(model, optimizer, train_loader)
    val_mse = test_rmse(model, val_loader)
    test_mse = test_rmse(model, test_loader)
    train_loss_hist.append(loss)
    val_loss_hist.append(val_mse)
    test_loss_hist.append(test_mse)
    scheduler.step(loss)
    if val_mse < best_val_mse:
        best_val_mse = val_mse # 105 epoch to get to 6 VAL RMSE
        mod_name = 'grove_KNN6_multi_GraphConv_2.pt' #  + "_" + str(epoch) + "_" +  str(best_val_mse) + '.pt'
        torch.save(model.state_dict(), mod_name) # 'model_inMemory_tr_test_minmax_noAug_K16_v2.pt'
        print(f'Epoch: {epoch:02d}, TR: {loss:.4f}, VAL: {val_mse:.4f}, TEST: {test_mse:.4f}')
        print("saving", mod_name)
    else:
        print(f'Epoch: {epoch:02d}, TR: {loss:.4f}, VAL: {val_mse:.4f}, TEST_MSE: {test_mse:.4f}')


# In[39]:


# loss graph

# Epoch: 419, TR: 13.0177, VAL: 14.3456, TEST: 11.8689, TEST_HUB: 9.6927
# saving grove_chance_edge_index_302_1.pt


x_list = [i for i in range(0, len(train_loss_hist))]

plt.plot(x_list, val_loss_hist, label='val')
plt.plot(x_list, train_loss_hist, label='train')

# plt.plot(x_list, test_loss_hist, label='test') # , color = 'red')

plt.legend()
plt.show()


# In[40]:


## load best model
device = torch.device('cuda') #  if torch.cuda.is_available() else 'cpu')

model_path = 'grove_KNN6_multi_GraphConv_2.pt' # '/home/scion.local/davidsos/Documents/DBH_Regression/model_19032022_StdS_conv3_dropout_10000_60_valAUG_MAE1.pt'
# good models: 'model_inMemory_tr_test_aug1446_K16_HUB_FPS_v1.pt'
MOD = GNN(hidden_channels=FILTERS).to(device)

MOD.load_state_dict(torch.load(model_path))
# MOD = torch.load(model_path)
# MOD

# model.state_dict()
# optimizer.state_dict()

# test_loader2 = DataLoader(DBH_DS, batch_size=32, shuffle = False)

# test_acc = test(MOD, test_loader2)
# print("TEST MAE", test_acc)


# In[41]:


## PREDICTIONS on a sample ##

@torch.no_grad()
def predict(model, DATA):
    narr = np.zeros(len(np.array(DATA.pos)[0:,0]), dtype = "int")
    narr = torch.from_numpy(narr).to(device)
#     for data in loader:
    model.eval()
    pred = model(DATA.pos.to(device), data.edge_index.to(device), narr)
#         y_output_pred.squeeze(-1)
#         loss = criterion(preds.to(torch.float32).squeeze(-1).to(device), data.y.to(torch.float32).to(device)) # Loss computation.
#         total_MSE += loss.item() #  * data.num_graphs
#         count+=1
        
    return DATA.chance.item(), pred[0].to(torch.float32).squeeze(-1).to(device).item(), DATA.ag.item(), pred[1].to(torch.float32).squeeze(-1).to(device).item() # len(test_loader.dataset))


# data_example = data_test[15]
# predict(model, data_example)


# In[66]:


DBH_DS_test[0].pos[0:, 0]


# In[ ]:





# In[42]:


## BATCH predict ##
import numpy as np

actual = []
pred = []
actual1 = []
pred1 = []

n_points = []

for data in DBH_DS_test:
    model.eval()
    y_true, y_hat, y_true1, y_hat1 = predict(MOD, data)
    n_points.append(len(data.pos[0:, 0]))
#     print(data.pos[0:, 0])
    actual.append(y_true)
    pred.append(y_hat)
    actual1.append(y_true1)
    pred1.append(y_hat1)

# inversed_actual = scaler.inverse_transform(np.array(actual).reshape(-1, 1))
# inversed_pred = scaler.inverse_transform(np.array(pred).reshape(-1, 1))

# rms = mean_squared_error(inversed_actual, inversed_pred, squared=False)
rms = mean_squared_error(actual, pred, squared=False)
MAPE = mean_absolute_percentage_error(actual, pred) # 20% (5.3, 0.15 on a good run!)

rms2 = mean_squared_error(actual1, pred1, squared=False)
MAPE2 = mean_absolute_percentage_error(actual1, pred1) # 20% (5.3, 0.15 on a good run!)

print(rms, MAPE, rms2, MAPE2)


# In[43]:


# y_df.describe()
# y_df.boxplot()
plt.boxplot([actual, pred])
# plt.boxplot([inversed_actual.flatten(), inversed_pred.flatten()])


# In[44]:


plt.boxplot([actual1, pred1])


# In[45]:


# plt.scatter(inversed_actual, inversed_pred)
plt.scatter(actual, pred)
plt.title("Param 1 - Branch Chance")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.show()


# In[46]:


slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(actual, pred)
print("R2", r_value)


# In[ ]:





# In[47]:


# plt.scatter(inversed_actual, inversed_pred)
plt.scatter(actual1, pred1)
plt.title("Param 2 - Chance Terminal")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


# In[48]:


slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(actual1, pred1)
print("R2", r_value)

