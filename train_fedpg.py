import argparse
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GATConv, GCN2Conv, GCNConv, SAGEConv, SGConv

from util.base_util import load_dataset, seed_everything

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()

# experimental environment setup
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--gpu_id', type=int, default=6)
parser.add_argument('--gpu_list', type=list, default=['1', '2', '3', '4', '5', '6'])
parser.add_argument('--root', type=str, default='/home/ubuntu/data/FEDPG')

# data setup
parser.add_argument('--task', type=str, default="node_cls", choices=["node_cls", "node_clustering", "link_pred"])
parser.add_argument('--dataset', type=str, default="Cora")
parser.add_argument('--partition', type=str,
                    default="Louvain", choices=["Louvain", "Metis"])
parser.add_argument('--part_delta', type=int, default=20)
parser.add_argument('--num_clients', type=int, default=3)

parser.add_argument('--model', type=str, default="gcn", choices=[
                    "gcn", "gat", "sgc", "gcn2", "gatv2", "ssgc", "graphsage", "gin", "lightgcn", "fagcn", "clustergcn", "appnp", "arma", "gae", "vgae"])
parser.add_argument('--optimizer', type=str, default="fedpg", choices=[
                    "fedavg", "fedprox", "scaffold", "moon", "fedtad", "feddc", "fedproto", "fednh", "fedpg"])
parser.add_argument('--num_rounds', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--num_part', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hid_dim', type=int, default=64)

parser.add_argument('--model_family', type=str, default=None, choices=[None, "family1"])

# fedpg 
parser.add_argument('--lambda_', type=float, default=5)
parser.add_argument('--lr_glb', type=float, default=1e-2)
parser.add_argument('--num_glb_epochs', type=int, default=10)
parser.add_argument('--glb_alpha', type=float, default=0.5)
parser.add_argument('--topk', type=float, default=2)
args = parser.parse_args()


class GCN(torch.nn.Module):

    def __init__(self, feat_dim, hid_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.unsup = False
        self.conv1 = GCNConv(feat_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = F.dropout(x, p=self.dropout)
        embedding = F.normalize(embedding, p=2, dim=1)
        x = self.conv2(embedding, edge_index)
        return embedding, x



class GAT(torch.nn.Module):

    def __init__(self, feat_dim, hid_dim, out_dim, dropout):
        super(GAT, self).__init__()
        self.unsup = False
        self.conv1 = GATConv(feat_dim, hid_dim)
        self.conv2 = GATConv(hid_dim, out_dim)
        self.dropout = dropout
        self.normalized= False

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = F.dropout(x, p=self.dropout)
        embedding = F.normalize(embedding, p=2, dim=1)
        x = self.conv2(embedding, edge_index)
        return embedding, x
    
    



class GraphSAGE(torch.nn.Module):

    def __init__(self, feat_dim, hid_dim, out_dim, dropout):
        super(GraphSAGE, self).__init__()
        self.unsup = False
        self.conv1 = SAGEConv(feat_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)
        self.dropout = dropout
        self.normalized= False

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = F.dropout(x, p=self.dropout)
        embedding = F.normalize(embedding, p=2, dim=1)
        x = self.conv2(embedding, edge_index)
        return embedding, x

  
  
class SGC(torch.nn.Module):

    def __init__(self, feat_dim, hid_dim, out_dim, dropout):
        super(SGC, self).__init__()
        self.unsup = False
        self.conv1 = SGConv(feat_dim, hid_dim)
        self.conv2 = SGConv(hid_dim, out_dim)
        self.dropout = dropout
        self.normalized= False

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = F.dropout(x, p=self.dropout)
        embedding = F.normalize(embedding, p=2, dim=1)
        x = self.conv2(embedding, edge_index)
        return embedding, x
  
  
class GCN2(torch.nn.Module):

    def __init__(self, feat_dim, hid_dim, out_dim, dropout):
        super(GCN2, self).__init__()
        self.unsup = False
        self.linear1 = nn.Linear(feat_dim, hid_dim)
        self.conv1 = GCN2Conv(hid_dim, alpha=0.1)
        self.conv2 = GCN2Conv(hid_dim, alpha=0.1)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)

        x0 = x
        x = self.conv1(x, x0, edge_index)
        embedding = self.conv2(x, x0, edge_index)
        embedding = F.normalize(embedding, p=2, dim=1)
        x = self.linear2(embedding)
        return embedding, x
    
def accuracy(pred, ground_truth):
    y_hat = pred.max(1)[1]
    correct = (y_hat == ground_truth).nonzero().shape[0]
    acc = correct / ground_truth.shape[0]
    return acc * 100


def tensor_to_list(input):
    tensor = input.nonzero().squeeze()
    if isinstance(tensor, torch.Tensor):
        if tensor.numel() == 1:
            return [tensor.item()]
        else:
            return tensor.tolist()
    else:
        raise TypeError("Input must be a PyTorch tensor.")


def neighbor_aware(data, num_classes):
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]
    
    
    # all edges
    neighbor = {}
    for i in range(num_nodes):
        neighbor[i] = set()
        
    for i in range(num_edges):
        source = int(data.edge_index[0, i])
        target = int(data.edge_index[1, i])
        
        if source == target:
            continue

        neighbor[source].add(target)
        neighbor[target].add(source)
    
    idx_dict = {class_i: {} for class_i in range(num_classes)}
    
    
    
    idx_dict = {
        class_i:{
            "self":None, # mask
            "1hop":{}, # {center: mask}
            "2hop":{}  # {center: mask}
            } for class_i in range(num_classes)
        
    }
    
    for class_i in range(num_classes):
        # self
        idx_dict[class_i]["self"] = (data.train_idx | data.val_idx) & (data.y == class_i)
        
        
        self_set = set(tensor_to_list(idx_dict[class_i]["self"]))
        
        # 1-hop
        for center in self_set:
            idx_dict[class_i]["1hop"][center] = torch.zeros_like(data.train_idx).long().to(data.train_idx.device)
            for next in neighbor[center]:
                idx_dict[class_i]["1hop"][center][next] = 1
        
        # 2-hop
        for center in self_set:
            idx_dict[class_i]["2hop"][center] = torch.zeros_like(data.train_idx).long().to(data.train_idx.device)
            hop1_set = tensor_to_list(idx_dict[class_i]["1hop"][center])
            for hop1_neighbor in hop1_set:
                for next in neighbor[hop1_neighbor]:
                    if next != center:
                        idx_dict[class_i]["2hop"][center][next] = 1
        

    return idx_dict
    

class Trainable_prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes * 3, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim), 
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        z = self.fc(mid)

        if z.dim() == 1:
            global_prototype = F.normalize(z.unsqueeze(0), p=2, dim=1)
        else:
            global_prototype = F.normalize(z, p=2, dim=1)
        return global_prototype


def class_hop_to_row_id(class_id, hop):
    if hop == "self":
        hop_id = 0
    elif hop == "1hop":
        hop_id = 1
    elif hop == "2hop":
        hop_id = 2
    
    return class_id * 3 + hop_id

def generate_global_prototype_dict(trainable_prototypes: Trainable_prototypes, num_classes):
        trainable_prototypes.eval()

        global_prototype_dict = {}
        for class_i in range(num_classes):
            for hop in ["self", "1hop", "2hop"]:
                row_id = class_hop_to_row_id(class_i, hop)
                
                if class_i not in global_prototype_dict:
                    global_prototype_dict[class_i] = {}
                global_prototype_dict[class_i][hop] = trainable_prototypes.forward(row_id).detach()
        return global_prototype_dict

seed_everything(seed=args.seed)
dataset = load_dataset(args)


if __name__ == "__main__":
    # start train
    personalized_global_prototypes = None

    CELoss = nn.CrossEntropyLoss()
    MSELoss = nn.MSELoss()
    device = torch.device(f"cuda:{args.gpu_id}")
    
    # client initialize
    subgraphs = [dataset.subgraphs[client_id].to(device) for client_id in range(args.num_clients)]
    if args.model_family is None:
        local_models = [ GCN(feat_dim=subgraphs[client_id].x.shape[1], 
                            hid_dim=args.hid_dim, 
                            out_dim=dataset.num_classes,
                            dropout=args.dropout).to(device)
                        for client_id in range(args.num_clients)]
    elif args.model_family == "family1":
        tot_models = [GCN, GAT, GraphSAGE, SGC, GCN2]
        tot_names = ["GCN", "GAT", "GraphSAGE", "SGC", "GCN2"]
        local_models = [ tot_models[int(5 * client_id / args.num_clients)](feat_dim=subgraphs[client_id].x.shape[1], 
                            hid_dim=args.hid_dim, 
                            out_dim=dataset.num_classes,
                            dropout=args.dropout).to(device)
                        for client_id in range(args.num_clients)]
        for client_id in range(args.num_clients):
            print(f"[client {client_id}] local model = {tot_names[int(5 * client_id / args.num_clients)]}")
        
    
    
    local_optimizers = [Adam(local_models[client_id].parameters(), lr=args.lr, weight_decay=args.weight_decay) 
                        for client_id in range(args.num_clients)]


    print("aware neighbor prototype...")
    l_idx_dict = []
    l_num_nodes = []
    for client_id in range(args.num_clients):
        idx_dict = neighbor_aware( subgraphs[client_id], dataset.num_classes)
        l_idx_dict.append(idx_dict)
        l_num_nodes.append(subgraphs[client_id].train_idx.sum())
    print("done")
    
    


    # server initialize
    trainable_prototypes = Trainable_prototypes(num_classes=dataset.num_classes,
                                                server_hidden_dim=args.hid_dim,
                                                feature_dim=args.hid_dim,
                                                device=device).to(device)
    
    global_optimizer = Adam(trainable_prototypes.parameters(), lr=args.lr_glb, weight_decay=5e-4)
    
    
    best_server_val = 0
    best_server_test = 0
    
    global_prototype_dict = None  


    for round_id in range(args.num_rounds):

        # local train
        for client_id in range(args.num_clients):
            for epoch_id in range(args.num_epochs):
                # set train mode
                local_models[client_id].train()
                local_optimizers[client_id].zero_grad()
                embedding, logits = local_models[client_id].forward(subgraphs[client_id])
                task_loss = CELoss(logits[subgraphs[client_id].train_idx], 
                               subgraphs[client_id].y[subgraphs[client_id].train_idx])


                prototype_align_loss = torch.tensor(0).float().to(device)
                
                if round_id != 0:
                    for class_i in range(dataset.num_classes):
                        
                        idx_dict = l_idx_dict[client_id][class_i]

                        
                        # 1. P(x|y)
                        if idx_dict["self"].any():
                            input = embedding[idx_dict["self"], :]
                            target = global_prototype_dict[client_id][class_i]["self"].expand_as(input)
                            prototype_align_loss += MSELoss(input, target)
                        else:
                            continue
                        
                        
                        # 2. P(x1|y)
                        embedding_1 = []
                        for center, hop1 in idx_dict["1hop"].items():
                            if hop1.any():
                                embedding_1.append(torch.mean(embedding[hop1, :], dim=0))
                            else:
                                continue
                            
                        if len(embedding_1) != 0:
                            input = torch.vstack(embedding_1)
                            target = global_prototype_dict[client_id][class_i]["1hop"].expand_as(input)
                            prototype_align_loss += MSELoss(input, target)

                        # 3. P(x2|y)  
                        embedding_2 = []
                        for center, hop2 in idx_dict["2hop"].items():
                            if hop2.any():
                                embedding_1.append(torch.mean(embedding[hop2, :], dim=0))
                            else:
                                continue
                        if len(embedding_2) != 0:
                            input = torch.vstack(embedding_2)
                            target = global_prototype_dict[client_id][class_i]["2hop"].expand_as(input)
                            prototype_align_loss += MSELoss(input, target)
                    
                loss_train = task_loss + args.lambda_ * prototype_align_loss
                loss_train.backward()
                local_optimizers[client_id].step()
        
        
        l_local_prototype_dict = []
        
        # generate local prototype
        
        for client_id in range(args.num_clients):
            local_models[client_id].eval()
            
            embedding, logits = local_models[client_id].forward(subgraphs[client_id])
            embedding = embedding.detach()
            local_prototype_dict = {class_i: {} for class_i in range(dataset.num_classes)}
        
            for class_i in range(dataset.num_classes):
                idx_dict = l_idx_dict[client_id][class_i]
                
                # self
                if idx_dict["self"].any():
                    local_prototype_dict[class_i]["self"] = torch.mean(embedding[idx_dict["self"], :], dim=0).squeeze()
                
                    # 1hop
                    embedding_1 = []
                    for center, hop1 in idx_dict["1hop"].items():
                        if hop1.any():
                            embedding_1.append(torch.mean(embedding[hop1, :], dim=0))
                        else:
                            continue
                        
                    if len(embedding_1) != 0:
                        local_prototype_dict[class_i]["1hop"] = torch.mean(torch.vstack(embedding_1), dim=0).squeeze()
                    
                    
                        embedding_2 = []
                        for center, hop2 in idx_dict["2hop"].items():
                            if hop2.any():
                                embedding_2.append(torch.mean(embedding[hop2, :], dim=0))
                            else:
                                continue
                        
                        if len(embedding_2) != 0:
                            local_prototype_dict[class_i]["2hop"] = torch.mean(torch.vstack(embedding_2), dim=0).squeeze()
                        else:
                            local_prototype_dict[class_i]["2hop"] = torch.zeros(embedding.size(1)).to(device)
                    
                    
                    else:
                        local_prototype_dict[class_i]["1hop"] = torch.zeros(embedding.size(1)).to(device)
                        local_prototype_dict[class_i]["2hop"] = torch.zeros(embedding.size(1)).to(device)
                
                else:
                    local_prototype_dict[class_i]["self"] = torch.zeros(embedding.size(1)).to(device)
                    local_prototype_dict[class_i]["1hop"] = torch.zeros(embedding.size(1)).to(device)
                    local_prototype_dict[class_i]["2hop"] = torch.zeros(embedding.size(1)).to(device)

                assert local_prototype_dict[class_i]["self"].dim() == 1
                assert local_prototype_dict[class_i]["1hop"].dim() == 1
                assert local_prototype_dict[class_i]["2hop"].dim() == 1
                
                
            l_local_prototype_dict.append(local_prototype_dict)
           
        # global aggregation
        trainable_prototypes.train()

        for hop in ["self", "1hop", "2hop"]:
            

            # w_list = []
            y_list = []
            tensor_list = []
            for client_i in range(args.num_clients):
                for class_i in range(dataset.num_classes):
                    y_list.append(class_i)
                    tensor_list.append(l_local_prototype_dict[client_i][class_i][hop]) 
            y = torch.tensor(y_list).type(torch.int64).to(device)
            local_prototypes = torch.cat([v.unsqueeze(0) for v in tensor_list], dim=0)
            row_id = [class_hop_to_row_id(class_id=class_id, hop=hop) for class_id in range(dataset.num_classes)]
               
            avg_proto = torch.zeros((dataset.num_classes, local_prototypes.shape[1])).to(device )
            num_local_prototypes = len(tensor_list) 
            for proto_i in range(num_local_prototypes):
                avg_proto[y_list[proto_i]] += local_prototypes[proto_i,:]
            for class_i in range(dataset.num_classes):
                avg_proto /= y_list.count(class_i)
                
                    
            gap = torch.ones(dataset.num_classes, device=device) * 1e9  
              

            for k1 in range(dataset.num_classes):
                for k2 in range(dataset.num_classes):
                    if k1 > k2:
                        dis = torch.norm(avg_proto[k1] - avg_proto[k2], p=2)
                        gap[k1] = torch.min(gap[k1], dis)
                        gap[k2] = torch.min(gap[k2], dis)
            min_gap = torch.min(gap)
            for i in range(len(gap)):
                if gap[i] > torch.tensor(1e8, device=device):
                    gap[i] = min_gap
            max_gap = torch.max(gap)

               
            for epoch in range(args.num_glb_epochs):
                global_optimizer.zero_grad()  
                global_prototypes = trainable_prototypes.forward(row_id)
                
                # fetch local prototypes in 'hop'
                features_square = torch.sum(torch.pow(local_prototypes, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(global_prototypes, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(local_prototypes, global_prototypes.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)
                
                one_hot = F.one_hot(y, dataset.num_classes).to(device)
                gap2 = min(max_gap.item(), 100)
                
                
                dist = dist + one_hot * gap2
                glb_loss = 0
                glb_loss = CELoss(-dist, y)

                glb_loss.backward()
                global_optimizer.step()
            
        # personalized finetune
        finetune_dict = {
            class_i:{
                hop:{
                    client_id:None
                    for client_id in range(args.num_clients)
                }
                for hop in ["self", "1hop", "2hop"]
            }
            for class_i in range(dataset.num_classes)
        }
        
        
        for class_i in range(dataset.num_classes):
            for hop in ["self", "1hop", "2hop"]:
                for center in range(args.num_clients):
                    similarity = []
                    for client_id in range(args.num_clients):
                        similarity.append((client_id, 
                                          torch.cosine_similarity(
                                                l_local_prototype_dict[center][class_i][hop],
                                                l_local_prototype_dict[client_id][class_i][hop],
                                                dim=0)
                                          ))
                    # find top-k similar clients
                    sorted_list = sorted(similarity, key=lambda x: x[1], reverse=True)
                    selected_list = [sorted_list[i][0] for i in range(args.topk)]
                    finetune_dict[class_i][hop][center] = selected_list

        trainable_prototypes.eval()
        
        global_prototype_dict = {
            client_id:{
                class_i:{
                    hop:None
                    for hop in ["self", "1hop", "2hop"]
                }
                for class_i in range(dataset.num_classes)
            }
            for client_id in range(args.num_clients)
        }
        
        
        for class_i in range(dataset.num_classes):
            for hop in ["self", "1hop", "2hop"]:
                row_id = class_hop_to_row_id(class_i, hop)
                global_prototype = trainable_prototypes.forward(row_id).detach()
                
                for center in range(args.num_clients):        
                    aggregated_prototype = torch.zeros_like(global_prototype).to(device)
                    for p in finetune_dict[class_i][hop][center]:
                        aggregated_prototype += (1-args.glb_alpha) * \
                        ( l_num_nodes[center] / sum([l_num_nodes[client_id] for client_id in finetune_dict[class_i][hop][center]])) * \
                        l_local_prototype_dict[p][class_i][hop]
                    aggregated_prototype += args.glb_alpha * global_prototype
                    aggregated_prototype /= 1 + len(finetune_dict[class_i][hop][center])

                    global_prototype_dict[center][class_i][hop] = aggregated_prototype
                    
        
        global_acc_val = 0
        global_acc_test = 0
        
        for client_id in range(args.num_clients):
            # set eval mode
            local_models[client_id].eval()
            embedding, logits = local_models[client_id].forward(subgraphs[client_id])

            acc_train = accuracy(logits[subgraphs[client_id].train_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].train_idx])
            acc_val = accuracy(logits[subgraphs[client_id].val_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].val_idx])
            acc_test = accuracy(logits[subgraphs[client_id].test_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].test_idx])
            
            print(f"[client {client_id}]: acc_train: {acc_train:.2f}\tacc_val: {acc_val:.2f}\tacc_test: {acc_test:.2f}")
            global_acc_val += subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] * acc_val
            global_acc_test += subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] * acc_test
            
        print(f"[server]: current_round: {round_id}\tglobal_val: {global_acc_val:.2f}\tglobal_test: {global_acc_test:.2f}")
        
        if global_acc_val > best_server_val:
            best_server_val = global_acc_val
            best_server_test = global_acc_test
            best_round = round_id
        print(f"[server]: best_round: {best_round}\tbest_val: {best_server_val:.2f}\tbest_test: {best_server_test:.2f}")
        print("-"*50)
            

    



