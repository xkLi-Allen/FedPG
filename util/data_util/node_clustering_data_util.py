import random

import torch
from torch_geometric.data import Data

from util.data_util.base_data_util import idx_to_mask


def construct_subgraph_dict_from_node_dict(
    num_clients, node_dict, G, graph_nx, train, val, test
):
    subgraph_list = []
    for client_id in range(num_clients):
        num_local_nodes = len(node_dict[client_id])
        print(f"num_local_nodes for c {client_id}: {num_local_nodes}")

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
        subgraph.num_samples = subgraph.num_nodes
        subgraph_list.append(subgraph)
        print(
            "Client: {}\tTotal Nodes: {}\tTotal Edges: {}".format(
                client_id + 1,
                subgraph.num_nodes,
                subgraph.num_edges,
            )
        )

    return subgraph_list
