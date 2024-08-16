import random

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from torch_geometric.data import Data
from torch_geometric.utils import (add_self_loops, from_scipy_sparse_matrix,
                                   to_dense_adj, to_scipy_sparse_matrix)

from util.data_util.base_data_util import idx_to_mask


def add_homophily_edges_efficient(subgraph, p):
    # 将 edge_index 转换为 CSR 格式的 scipy 稀疏矩阵
    edge_index = subgraph.edge_index
    num_nodes = subgraph.num_nodes
    adj_matrix_sparse = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    
    # 获取节点的标签
    y = subgraph.y.numpy()
    
    # 对于每个 unique label, 找到对应的节点 indices
    for label in np.unique(y):
        node_indices = np.where(y == label)[0]
        
        # 对于这些节点，我们创建一个完全连接的子图（使用概率 p）
        for i in node_indices:
            for j in node_indices:
                if i != j and np.random.rand() < p:
                    # 只有当 i, j 之间没有现有边时才添加边
                    if not adj_matrix_sparse[i, j]:
                        adj_matrix_sparse[i, j] = 1
                        adj_matrix_sparse[j, i] = 1  # 确保无向图的性质
    
    # 将 CSR 稀疏矩阵转换回 PyG 的 edge_index
    edge_index_updated = from_scipy_sparse_matrix(adj_matrix_sparse)
    
    # 更新图
    subgraph.edge_index = edge_index_updated[0]
    return subgraph


def construct_subgraph_dict_from_node_dict(
    num_clients, node_dict, G, graph_nx, train, val, test, add_random_homo_edges=False
):
    subgraph_list = []
    for client_id in range(num_clients):
        num_local_nodes = len(node_dict[client_id])
        print(f"num_local_nodes for c {client_id}: {num_local_nodes}")
        
        train_idx = []
        val_idx = []
        test_idx = []
        class_i_idx_list = {}
        
        for idx in range(num_local_nodes):
            label = G.y[node_dict[client_id][idx]]
            if int(label) not in class_i_idx_list:
                class_i_idx_list[int(label)] = []
            class_i_idx_list[int(label)].append(idx)
        
        for class_i in range(G.num_classes):
            if class_i not in class_i_idx_list:
                continue
            local_node_idx = class_i_idx_list[class_i]
            random.shuffle(local_node_idx)

            # 在每一个类别中随机抽 train/val/test, 保证 train 和 test 同分布
            num_local_nodes_class_i = len(local_node_idx)
            train_size = int(num_local_nodes_class_i * train)
            val_size = int(num_local_nodes_class_i * val)
            test_size = int(num_local_nodes_class_i * test) 

            train_idx += local_node_idx[:train_size]
            val_idx += local_node_idx[train_size: train_size + val_size]
            test_idx += local_node_idx[train_size + val_size:]

        assert len(train_idx) + len(val_idx) + len(test_idx) == num_local_nodes
        # store local (client i) node index in node_dict[client i]
        local_train_idx = idx_to_mask(train_idx, size=num_local_nodes)
        local_val_idx = idx_to_mask(val_idx, size=num_local_nodes)
        local_test_idx = idx_to_mask(test_idx, size=num_local_nodes)




        # store mapping relationship
        node_idx_map = {}
        edge_idx = []
        for idx in range(num_local_nodes):
            node_idx_map[node_dict[client_id][idx]] = idx

        # obtain induced subgraphs
        edge_idx += [
            (node_idx_map[x[0]], node_idx_map[x[1]])
            for x in graph_nx.subgraph(node_dict[client_id]).edges
        ]
        edge_idx += [
            (node_idx_map[x[1]], node_idx_map[x[0]])
            for x in graph_nx.subgraph(node_dict[client_id]).edges
        ]

        edge_idx_tensor = torch.tensor(edge_idx, dtype=torch.long).T
        subgraph = Data(
            x=G.x[node_dict[client_id]],
            y=G.y[node_dict[client_id]],
            edge_index=edge_idx_tensor,
        )


        if add_random_homo_edges:
            subgraph = add_homophily_edges_efficient(subgraph, p=0.01)


        subgraph.train_idx = local_train_idx
        subgraph.val_idx = local_val_idx
        subgraph.test_idx = local_test_idx
        subgraph.num_samples = subgraph.num_nodes
        subgraph_list.append(subgraph)
        print(
            "Client: {}\tTotal Nodes: {}\tTotal Edges: {}\tTrain Nodes: {}\tVal Nodes: {}\tTest Nodes\t{}".format(
                client_id + 1,
                subgraph.num_nodes,
                subgraph.num_edges,
                len(train_idx),
                len(val_idx),
                len(test_idx),
            )
        )

    return subgraph_list
