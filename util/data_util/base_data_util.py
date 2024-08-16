import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch_geometric.utils.convert import to_networkx

from louvain.community import community_louvain


def remove_self_loops(edge_index, edge_attr=None):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def to_undirected(edge_index):
    if isinstance(edge_index, sp.csr_matrix) or isinstance(edge_index, sp.coo_matrix):
        row, col = edge_index.row, edge_index.col
        row, col = torch.from_numpy(row), torch.from_numpy(col)
    else:
        row, col = edge_index
        if not isinstance(row, Tensor) or not isinstance(col, Tensor):
            row, col = torch.from_numpy(row), torch.from_numpy(col)
    new_row = torch.hstack((row, col))
    new_col = torch.hstack((col, row))
    new_edge_index = torch.stack((new_row, new_col), dim=0)
    return new_edge_index


def idx_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def louvain_partition(graph, num_clients, delta=20):
    num_nodes = graph.number_of_nodes()

    partition = community_louvain.best_partition(graph)

    groups = []

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    print(groups)
    partition_groups = {group_i: [] for group_i in groups}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)

    group_len_max = num_nodes // num_clients - delta
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    print(groups)

    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict = {}

    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {
        k: v
        for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)
    }

    owner_node_ids = {owner_id: [] for owner_id in range(num_clients)}

    owner_nodes_len = num_nodes // num_clients
    owner_list = [i for i in range(num_clients)]
    owner_ind = 0

    bad_key = 1000

    for group_i in sort_len_dict.keys():
        while (
            len(owner_list) >= 2
            and len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len
        ):
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        cnt = 0
        while (
            len(owner_node_ids[owner_list[owner_ind]]) +
                len(partition_groups[group_i])
            >= owner_nodes_len + delta
        ):
            owner_ind = (owner_ind + 1) % len(owner_list)
            cnt += 1
            if cnt > bad_key:
                cnt = 0
                min_v = 1e15
                for i in range(len(owner_list)):
                    if len(owner_node_ids[owner_list[owner_ind]]) < min_v:
                        min_v = len(owner_node_ids[owner_list[owner_ind]])
                        owner_ind = i
                break

        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
    node_dict = owner_node_ids

    print("end louvain")
    return node_dict


def data_partition(task, G, num_clients, train, val, test, partition, part_delta):
    time_st = time.time()
    print("start g to nxg")
    graph_nx = to_networkx(G, to_undirected=True, remove_self_loops=True)
    print(f"get nxg {time.time()-time_st} sec.")

    if partition == "Louvain":
        print("Conducting louvain graph partition...")
        node_dict = louvain_partition(
            graph=graph_nx, num_clients=num_clients, delta=part_delta
        )
    elif partition == "Metis":
        # import metispy as metis
        import metis

        print("Conducting metis graph partition...")
        node_dict = {}
        n_cuts, membership = metis.part_graph(graph_nx, num_clients)
        for client_id in range(num_clients):
            client_indices = np.where(np.array(membership) == client_id)[0]
            client_indices = list(client_indices)
            node_dict[client_id] = client_indices
    else:
        raise ValueError(f"No such partition method: '{partition}'.")

    assert sum([len(node_dict[i])
               for i in range(len(node_dict))]) == G.num_nodes
    
    if task == "node_cls":
        from util.data_util.node_cls_data_util import \
            construct_subgraph_dict_from_node_dict
    elif task == "node_clustering":
        from util.data_util.node_clustering_data_util import \
            construct_subgraph_dict_from_node_dict  
    elif task == "link_pred":
        from util.data_util.link_pred_data_util import \
            construct_subgraph_dict_from_node_dict
            
            
    subgraph_list = construct_subgraph_dict_from_node_dict(
        G=G,
        num_clients=num_clients,
        node_dict=node_dict,
        graph_nx=graph_nx,
        train=train,
        val=val,
        test=test,
    )
    
    G.num_samples = 0
    for subgraph in subgraph_list:
        G.num_samples += subgraph.num_samples
    return subgraph_list


def analysis_graph_structure_homo_hete_info(G):
    structure_homo_hete_label_info = {}
    structure_homo_hete_label_info["node_homophily"] = label_node_homogeneity(
        G)
    structure_homo_hete_label_info["edge_homophily"] = label_edge_homogeneity(
        G)
    print(
        f"homo_node: {structure_homo_hete_label_info['node_homophily']:.4f}\nhomo_edge: {structure_homo_hete_label_info['edge_homophily']:.4f}"
    )
    return (
        structure_homo_hete_label_info["node_homophily"],
        structure_homo_hete_label_info["edge_homophily"],
    )


def label_node_homogeneity(G):
    num_nodes = G.num_nodes
    homophily = 0
    for edge_u in range(num_nodes):
        hit = 0
        edge_v_list = G.edge_index[1][torch.where(G.edge_index[0] == edge_u)]
        if len(edge_v_list) != 0:
            for i in range(len(edge_v_list)):
                edge_v = edge_v_list[i]
                if G.y[edge_u] == G.y[edge_v]:
                    hit += 1
            homophily += hit / len(edge_v_list)
    homophily /= num_nodes
    return homophily


def label_edge_homogeneity(G):
    num_edges = G.num_edges
    homophily = 0
    for i in range(num_edges):
        if G.y[G.edge_index[0][i]] == G.y[G.edge_index[1][i]]:
            homophily += 1
    homophily /= num_edges
    return homophily
