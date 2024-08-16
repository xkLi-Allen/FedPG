## README

**Requirements**

Hardware environment: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz, NVIDIA GeForce RTX 3090 with 24GB memory.

Software environment: Ubuntu 20.04.1, Python 3.9, PyTorch 1.13.1 and CUDA 11.3

1. Please refer to [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install the environments;
2. Run 'pip install -r requirements.txt' to download required packages;



**Training**

To train the model(s) in the paper

1. Please unzip `Cora.zip` to your data path.
2. Change the `--root` argument in the `main.py` to your data path.
3. Run `main.py` to train GCN with our proposed federated graph model optimization strategies (FedPG). 



**Performance Highlights**

Here we present some examples of training FedPG:

1. Same Backbone (GCN) on Cora-Louvain-10 Clients

```python
python main.py --dataset Cora --num_clients 10 --partition Louvain --lambda_ 9.3351 --num_glb_epochs 10 --lr_glb 0.01 --glb_alpha 0.4029 --topk 2
```

​	

2. Same Backbone (GCN) on Cora-Metis-10 Clients

```python
python main.py --dataset Cora --num_clients 10 --partition Metis --lambda_ 9.3351 --num_glb_epochs 10 --lr_glb 0.01 --glb_alpha 0.4029 --topk 2
```



3. Heterogeneous Backbones (GCN, GAT, GraphSAGE, SGC, GCN2) on Cora-Louvain-10 Clients

```python
python main.py --dataset Cora --num_clients 10 --partition Louvain --lambda_ 9.8333 --num_glb_epochs 3 --lr_glb 0.01 --glb_alpha 0.3831 --topk 2 --model_family family1
```

​	

4. Heterogeneous Backbones (GCN, GAT, GraphSAGE, SGC, GCN2) on Cora-Metis-10 Clients

```python
python main.py --dataset Cora --num_clients 10 --partition Metis --lambda_ 5.2815 --num_glb_epochs 50 --lr_glb 0.001 --glb_alpha 0.7257 --topk 2 --model_family family1
```

​	
