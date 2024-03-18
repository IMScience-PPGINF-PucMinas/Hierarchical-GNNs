import torch
import numpy as np
import mlflow.pytorch
from torch_geometric.data import Data
from dataset_creation import MG_Cifar10_graphs
from sklearn.metrics import  accuracy_score
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from networkx import draw
from model import HIGSI
from torch_geometric.utils import degree
import sys

RAMDOM_SEED=int(sys.argv[1])

torch.cuda.manual_seed(RAMDOM_SEED)
torch.manual_seed(RAMDOM_SEED)
np.random.seed(RAMDOM_SEED)

NUM_CLASSES=10
BATCH_SIZE=64
EPOCHS=300

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("HIGSI - Target 20 nodes")

class MultiGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'reduced_index':
            return 1
        if key == 'graph_index':
            return self.ng
        return super().__inc__(key, value, *args, **kwargs)


def plot_graph(data):
    g = to_networkx(data, to_undirected=False)
    draw(g)
    print("")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def val(epoch, model, val_loader, loss_fn, device, key):
    all_preds=[]
    all_labels=[]
    model.eval()
    for batch in val_loader:
        batch.to(device)
        pred = model(x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch_index=batch.batch,
                    graph_index=batch.graph_index, 
                    reduced_index=batch.reduced_index)
        #loss = loss_fn(pred, torch.argmax(batch.y.reshape(len(batch),10), dim=1)) #torch.argmax(batch.y.reshape(len(batch),10), dim=1)
        loss = loss_fn(pred, batch.y.reshape(len(batch),10))
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(np.argmax(batch.y.reshape(len(batch),10).cpu().detach().numpy(), axis=1))
    all_preds=np.concatenate(all_preds).ravel()
    all_labels=np.concatenate(all_labels).ravel()
    accu = calculate_metrics(all_preds, all_labels, epoch, key)
    return loss, accu

def calculate_metrics (y_pred , y_true , epoch, type) :
    print (f"{type} Accuracy:{accuracy_score(y_pred, y_true)}")
    mlflow.log_metric(key=f"Accuracy-{type}", value=float(accuracy_score(y_pred, y_true)), step=epoch)
    return accuracy_score(y_pred, y_true)

def train(epoch, train_loader, device, optimizer, model, loss_fn):
    all_preds = []
    all_labels = []
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        pred=model(x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch_index=batch.batch,
                    graph_index=batch.graph_index, 
                    reduced_index=batch.reduced_index)
        # loss = loss_fn(pred, torch.argmax(batch.y.reshape(len(batch),10), dim=1))
        loss = loss_fn(pred, batch.y.reshape(len(batch),10))
        loss.backward()
        optimizer.step()
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(np.argmax(batch.y.reshape(len(batch),10).cpu().detach().numpy(), axis=1))
    all_preds=np.concatenate(all_preds).ravel()
    all_labels=np.concatenate(all_labels).ravel()
    accu = calculate_metrics(all_preds, all_labels, epoch, "train")
    return loss, accu, optimizer

def main():
    # torch.cuda.set_per_process_memory_fraction(fraction=0.8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = MG_Cifar10_graphs(root="data/", nodes=20, canonized=True, split='train') 
    val_data = MG_Cifar10_graphs(root="data/", nodes=20, canonized=True, split='val')
    test_data = MG_Cifar10_graphs(root="data/", nodes=20, canonized=True, split='test')
             

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = HIGSI(node_feature_size=train_data[0].x.shape[1], edge_feature_size=train_data[0].edge_attr.shape[1], num_classes=train_data[0].y.shape[0])

    model = model.to(device)                                    
    print(f"Number of parameters: {count_parameters(model)}") 
    
    save_dir = f"weights/HIGSI/3_blocks_20_nodes_seed={RAMDOM_SEED}.pth"
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=3e-6)                                                                               
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, min_lr=1e-6, factor=0.5, verbose=True)
    best_accu=0
    print("-----------------------------------------------------------")
    with mlflow.start_run() as run:
        for epoch in range(EPOCHS):
            model.train()
            loss, accu, optimizer = train(epoch=epoch, train_loader=train_loader, device=device, optimizer=optimizer, model=model, loss_fn=loss_fn)           
            print(f"Epoch {epoch+1} | Train Loss {loss}")
            mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)
            print("--")
            
            ##### Validation
            model.eval()
            epoch_val_loss, epoch_val_accu = val(epoch=epoch, model=model, val_loader=val_loader, device=device, loss_fn=loss_fn, key="val")
            print(f"Epoch {epoch+1} | Val Loss {epoch_val_loss}")
            print("--")
            _ , epoch_test_accu = val(epoch=epoch, model=model, val_loader=test_loader, device=device, loss_fn=loss_fn, key="test")
            print(f"Epoch {epoch+1} | Test Accu {epoch_test_accu}")
            print("-----------------------------------------------------------")
            mlflow.log_metric(key="val loss", value=float(epoch_val_loss), step=epoch)
            if(epoch_val_accu > best_accu):
                best_accu = epoch_val_accu
                torch.save(model.state_dict(), save_dir)
            scheduler.step(epoch_val_accu) 

    print("Train Done")


if __name__ == "__main__":
    main()