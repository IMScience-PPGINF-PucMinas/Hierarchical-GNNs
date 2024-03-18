import torch
import numpy as np
import torch.nn as nn
from torch_geometric.data import Data
from model import HIGSI
from sklearn.metrics import accuracy_score
from dataset_creation import MG_Cifar10_graphs
from torch_geometric.loader import DataLoader
import sys

RAMDOM_SEED=int(sys.argv[1])

torch.cuda.manual_seed(RAMDOM_SEED)
torch.manual_seed(RAMDOM_SEED)
np.random.seed(RAMDOM_SEED)


BATCH_SIZE=64
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics (y_pred , y_true , type="test") :
    print (f"Accuracy:{accuracy_score(y_pred, y_true)}")
   
def main():
    test_set = MG_Cifar10_graphs(root="data/", nodes=20, canonized=True, split='test')  
     
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HIGSI(node_feature_size=test_set[0].x.shape[1], edge_feature_size=test_set[0].edge_attr.shape[1], num_classes=test_set[0].y.shape[0])
    
    print(f"Number of parameters: {count_parameters(model)}")   #Parameters of the model
    model.load_state_dict(torch.load(f"weights/HIGSI/3_blocks_20_nodes_seed={RAMDOM_SEED}.pth"))
    model = model.to(device)
    model.eval()

    all_preds=[]
    all_labels=[]
    for batch in test_loader:
        batch.to(device)
        pred = model(x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch_index=batch.batch,
                    graph_index=batch.graph_index, 
                    reduced_index=batch.reduced_index)
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(np.argmax(batch.y.reshape(len(batch),10).cpu().detach().numpy(), axis=1))
        
    all_preds=np.concatenate(all_preds).ravel()
    all_labels=np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, "test")

if __name__ == "__main__":
    main()