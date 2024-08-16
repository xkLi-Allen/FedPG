import random

import torch
from torch_geometric.data import Data

from util.data_util.base_data_util import idx_to_mask

# 链路预测需要保留的是边的对应关系
# 每一个 sample 为一条边, 它的 label 为 0 或 1 (采样负边或正边)
# local graph 学习时, 本地 GNN 对每一个节点输出 embedding, 然后得到各个边对应的概率
# global data 实际上不需要 ?


def construct_subgraph_dict_from_node_dict(
    num_clients, node_dict, G, graph_nx, train, val, test
):
    subgraph_list = []
    
    for client_id in range(num_clients):
        num_local_nodes = len(node_dict[client_id])
        
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

        edge_idx_tensor = torch.tensor(edge_idx, dtype=torch.long).T # 全部正样本的 edge_index
        
        subgraph = Data(
            x=G.x[node_dict[client_id]],
            y=G.y[node_dict[client_id]],
            edge_index=edge_idx_tensor,
        )
        
        # 划分数据集 ( 目前的 edge_index 中只有正边, 因为每一次训练时需要重新采样负边, 但 edge_index 的 idx 和 label 可提前固定)
        num_local_edges = edge_idx_tensor.shape[1] # 边的数量
        
        local_edge_idx = [idx for idx in range(num_local_edges)]
        random.shuffle(local_edge_idx)

        train_size = int(num_local_edges * train)
        val_size = int(num_local_edges * val)
        test_size = int(num_local_edges * test)

        train_idx = local_edge_idx[:train_size]
        val_idx = local_edge_idx[train_size: train_size + val_size]
        test_idx = local_edge_idx[train_size + val_size:]

        assert len(train_idx) + len(val_idx) + len(test_idx) == num_local_edges
        
        # store local (client i) node index in node_dict[client i]
        
        # 负边个数与正边个数相同
        num_local_edges = num_local_edges * 2
        
        
        pos_train_edge_index_idx = idx_to_mask(train_idx, size=num_local_edges)
        pos_val_edge_index_idx = idx_to_mask(val_idx, size=num_local_edges)
        pos_test_edge_index_idx = idx_to_mask(test_idx, size=num_local_edges)
        
        neg_train_edge_index_idx = idx_to_mask([i + num_local_edges//2 for i in train_idx], size=num_local_edges)
        neg_val_edge_index_idx = idx_to_mask([i + num_local_edges//2 for i in val_idx], size=num_local_edges)
        neg_test_edge_index_idx = idx_to_mask([i + num_local_edges//2 for i in test_idx], size=num_local_edges)


        subgraph.pos_train_edge_index_idx = pos_train_edge_index_idx
        subgraph.pos_val_edge_index_idx = pos_val_edge_index_idx
        subgraph.pos_test_edge_index_idx = pos_test_edge_index_idx
        subgraph.neg_train_edge_index_idx = neg_train_edge_index_idx
        subgraph.neg_val_edge_index_idx = neg_val_edge_index_idx
        subgraph.neg_test_edge_index_idx = neg_test_edge_index_idx
        subgraph.edge_label = torch.cat((torch.ones(size=(num_local_edges // 2,)), 
                                         torch.zeros(size=(num_local_edges // 2,))), dim=0).squeeze().cpu()
        subgraph.num_samples = num_local_edges

        subgraph_list.append(subgraph)
        print(
            "Client: {}\tTotal Nodes: {}\tTotal Edges: {}\tTrain Nodes: {}\tVal Nodes: {}\tTest Nodes\t{}".format(
                client_id + 1,
                subgraph.num_nodes,
                subgraph.num_edges,
                train_size,
                val_size,
                test_size,
            )
        )

    return subgraph_list
