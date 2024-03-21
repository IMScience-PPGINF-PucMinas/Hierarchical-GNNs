import torch
import torch_geometric
from torchvision.datasets import CIFAR10, STL10
from torchvision.transforms import ToTensor, Lambda
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
import graph_utils
from tqdm import tqdm
# torch.manual_seed(1)
# np.random.seed(1)


print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class Cifar10_graphs(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, split='train',  nodes=100):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.split = split
        self.nodes = nodes
        if(self.split == 'train'):
            self.train_indices=np.load("train_indices-cifar.npy")
            self.data = CIFAR10("data/raw/", download=True, transform=graph_utils.be_np, 
                target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        elif(self.split == 'val'):
            self.val_indices=np.load("val_indices-cifar.npy")
            self.data = CIFAR10("data/raw/", download=True, transform=graph_utils.be_np, 
                target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))   
        elif(self.split == 'test'):
            self.data = CIFAR10("data/raw/", download=True, transform=graph_utils.be_np, train=False,
                target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        super(Cifar10_graphs, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return 'cifar-10-python.tar.gz'

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        print("Using Superpixel RAG!")
        if(self.split == 'train'):
            return [f'RAG_train_{i}.pt' for i in range(self.train_indices.shape[0])]
        elif(self.split == 'val'):
            return [f'RAG_val_{i}.pt' for i in range(self.val_indices.shape[0])]
        else:
            return [f'RAG_test_{i}.pt' for i in range(len(self.data))]
        
    def download(self):
        pass

    def process(self):
        print("Using superpixel RAG target "+str(self.nodes)+" number of nodes") 
        if(self.split == 'test'):
            for i in tqdm(range(len(self.data))):
                #print(i)
                node_features, coo, edge_features, pos = graph_utils.RAG(self.data[i][0], n_nodes=self.nodes)
                data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                            edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                            edge_attr=torch.from_numpy(edge_features),
                            y=self.data[i][1],
                            pos=torch.from_numpy(pos))
                path = os.path.join('data/processed/', f'RAG_test_{i}.pt')
                torch.save(data, path)
        elif(self.split == 'val'):
            j=0
            for i in tqdm(self.val_indices):
                #print(i)
                node_features, coo, edge_features, pos = graph_utils.RAG(self.data[i][0], n_nodes=self.nodes)
                data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                            edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                            edge_attr=torch.from_numpy(edge_features),
                            y=self.data[i][1],
                            pos=torch.from_numpy(pos))
                path = os.path.join('data/processed/', f'RAG_val_{j}.pt')
                torch.save(data, path)
                j+=1
        elif(self.split == 'train'):
            j=0
            for i in tqdm(self.train_indices):
                node_features, coo, edge_features, pos = graph_utils.RAG(self.data[i][0], n_nodes=self.nodes)
                data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                            edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                            edge_attr=torch.from_numpy(edge_features),
                            y=self.data[i][1],
                            pos=torch.from_numpy(pos))
                path = os.path.join('data/processed/', f'RAG_train_{j}.pt')
                torch.save(data, path)
                j+=1
           
    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        if(self.split == 'train'):
            return self.train_indices.shape[0]
        elif(self.split == 'val'):
            return self.val_indices.shape[0]
        else:
            return len(self.data)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if(self.split == 'train'):
            data = torch.load(os.path.join(self.processed_dir, f'RAG_train_{idx}.pt'))
        elif(self.split == 'val'):
            data = torch.load(os.path.join(self.processed_dir, f'RAG_val_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'RAG_test_{idx}.pt'))        
        return data
            
class STL10_graphs(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, split='train', nodes=100, hierarchy=True, knn=False, k_neighbors=8, complete=False, include_rag=False, include_cuts=False, canonized=False, multi_rag=False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.split = split
        self.nodes = nodes
        self.hierarchy = hierarchy 
        self.knn = knn 
        self.complete = complete
        self.rag = include_rag
        self.cuts = include_cuts
        self.k_neighbors = k_neighbors
        self.canonized = canonized
        self.multi_RAG = multi_rag

        if(self.split == 'train'):
            self.train_indices=np.load("train_indices-stl.npy")
            
            self.data = STL10("data/raw/", download=True, transform=graph_utils.be_np,split='train',
                        target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        elif(self.split == 'val'):
            self.val_indices=np.load("val_indices-stl.npy")

            self.data = STL10("data/raw/", download=True, transform=graph_utils.be_np, split='train', 
                        target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))   
        else:
            self.data = STL10("data/raw", download=True, transform=graph_utils.be_np, split='test',
                        target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

        super(STL10_graphs, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return 'stl10_binary.tar.gz'

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        if(not self.hierarchy):
            if(self.multi_RAG):
                print("Using Superpixel MULTI RAG!")
                if(self.split == 'train'):
                    return [f'MULTI_RAG_train_{i}.pt' for i in range(self.train_indices.shape[0])]
                elif(self.split == 'val'):
                    return [f'MULTI_RAG_val_{i}.pt' for i in range(self.val_indices.shape[0])]
                else:
                    return [f'MULTI_RAG_test_{i}.pt' for i in range(len(self.data))]
            else:    
                print("Using Superpixel RAG!")
                if(self.split == 'train'):
                    return [f'RAG_train_{i}.pt' for i in range(self.train_indices.shape[0])]
                elif(self.split == 'val'):
                    return [f'RAG_val_{i}.pt' for i in range(self.val_indices.shape[0])]
                else:
                    return [f'RAG_test_{i}.pt' for i in range(len(self.data))]
        else:
            if(self.complete and self.knn):
                load_str = 'complete_hierarchy'
            elif(self.knn):   
                load_str = 'knn_hierarchy'
            else:
                if(self.cuts and self.rag): 
                    load_str = 'hierarchy+RAG+CUTS'
                elif(self.cuts):
                    load_str = 'hierarchy+CUTS'
                elif(self.rag):
                    load_str = 'hierarchy+RAG'
                else:
                    load_str = 'hierarchy'
            if(self.split == 'train'):
                return [load_str+f'_train_{i}.pt' for i in range(self.train_indices.shape[0])]
            elif(self.split == 'val'):
                return [load_str+f'_val_{i}.pt' for i in range(self.val_indices.shape[0])]
            else:
                return [load_str+f'_test_{i}.pt' for i in range(len(self.data))]
            
                  
    def download(self):
        pass

    def process(self):
        if(self.hierarchy):
            if(self.complete and self.knn):
                print("Using hierarchy target "+str(self.nodes)+" number of nodes COMPLETE ADJ.")
                save_str = 'complete_hierarchy'
            elif(self.knn):
                print("Using hierarchy target "+str(self.nodes)+" number of nodes "+str(self.k_neighbors)+"-NN ADJ.")
                save_str = 'knn_hierarchy'
            else:
                if(self.cuts and self.rag):
                    print("Using hierarchy target "+str(self.nodes)+" superpixels HIERARCHY+RAG+CUTS ADJ.")
                    save_str = 'hierarchy+RAG+CUTS'
                elif(self.cuts):
                    print("Using hierarchy target "+str(self.nodes)+" superpixels HIERARCHY+CUTS ADJ.")
                    save_str = 'hierarchy+CUTS'
                elif(self.rag):
                    print("Using hierarchy target "+str(self.nodes)+" superpixels HIERARCHY+RAG ADJ.")
                    save_str = 'hierarchy+RAG'
                else:
                    print("Using hierarchy target "+str(self.nodes)+" superpixels HIERARCHY ADJ.")
                    save_str = 'hierarchy'
            if(self.split == 'test'):
                for i in tqdm(range(len(self.data))):
                    #print(i)
                    node_features, coo, edge_features, pos = graph_utils.superpixel_hierarchy(self.data[i][0], n_nodes=self.nodes, knn=self.knn,
                                                                                            k_neighbors=self.k_neighbors, complete=self.complete, rag=self.rag,
                                                                                            cuts_adj=self.cuts, canonized=self.canonized)
                    data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos))
                    path = os.path.join('data/processed/', save_str+f'_test_{i}.pt')
                    torch.save(data, path)
            elif(self.split == 'val'):
                j=0
                for i in tqdm(self.val_indices):
                    #print(i)
                    node_features, coo, edge_features, pos = graph_utils.superpixel_hierarchy(self.data[i][0], n_nodes=self.nodes, knn=self.knn,
                                                                                            k_neighbors=self.k_neighbors, complete=self.complete, rag=self.rag,
                                                                                            cuts_adj=self.cuts, canonized=self.canonized)
                    data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos))
                    path = os.path.join('data/processed/', save_str+f'_val_{j}.pt')
                    torch.save(data, path)
                    j+=1
            elif(self.split == 'train'):
                j=0
                for i in tqdm(self.train_indices):
                    node_features, coo, edge_features, pos = graph_utils.superpixel_hierarchy(self.data[i][0], n_nodes=self.nodes, knn=self.knn,
                                                                                            k_neighbors=self.k_neighbors, complete=self.complete, rag=self.rag,
                                                                                            cuts_adj=self.cuts, canonized=self.canonized)
                    data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos))
                    path = os.path.join('data/processed/', save_str+f'_train_{j}.pt')
                    torch.save(data, path)
                    j+=1
        if(not self.hierarchy):
            if(self.multi_RAG):
                print("Using superpixel MULTI RAG target "+str(self.nodes)+" number of nodes") 
                if(self.split == 'test'):
                    for i in tqdm(range(len(self.data))):
                        #print(i)
                        node_features, coo, edge_features, pos = graph_utils.multi_RAG(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                        data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                    edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                    edge_attr=torch.from_numpy(edge_features),
                                    y=self.data[i][1],
                                    pos=torch.from_numpy(pos))
                        path = os.path.join('data/processed/', f'MULTI_RAG_test_{i}.pt')
                        torch.save(data, path)
                elif(self.split == 'val'):
                    j=0
                    for i in tqdm(self.val_indices):
                        #print(i)
                        node_features, coo, edge_features, pos = graph_utils.multi_RAG(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                        data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                    edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                    edge_attr=torch.from_numpy(edge_features),
                                    y=self.data[i][1],
                                    pos=torch.from_numpy(pos))
                        path = os.path.join('data/processed/', f'MULTI_RAG_val_{j}.pt')
                        torch.save(data, path)
                        j+=1
                elif(self.split == 'train'):
                    j=0
                    for i in tqdm(self.train_indices):
                        node_features, coo, edge_features, pos = graph_utils.multi_RAG(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                        data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                    edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                    edge_attr=torch.from_numpy(edge_features),
                                    y=self.data[i][1],
                                    pos=torch.from_numpy(pos))
                        path = os.path.join('data/processed/', f'MULTI_RAG_train_{j}.pt')
                        torch.save(data, path)
                        j+=1
            else:   
                print("Using superpixel RAG target "+str(self.nodes)+" number of nodes") 
                if(self.split == 'test'):
                    for i in tqdm(range(len(self.data))):
                        #print(i)
                        node_features, coo, edge_features, pos = graph_utils.RAG(self.data[i][0], n_nodes=self.nodes)
                        data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                    edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                    edge_attr=torch.from_numpy(edge_features),
                                    y=self.data[i][1],
                                    pos=torch.from_numpy(pos))
                        path = os.path.join('data/processed/', f'RAG_test_{i}.pt')
                        torch.save(data, path)
                elif(self.split == 'val'):
                    j=0
                    for i in tqdm(self.val_indices):
                        #print(i)
                        node_features, coo, edge_features, pos = graph_utils.RAG(self.data[i][0], n_nodes=self.nodes)
                        data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                    edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                    edge_attr=torch.from_numpy(edge_features),
                                    y=self.data[i][1],
                                    pos=torch.from_numpy(pos))
                        path = os.path.join('data/processed/', f'RAG_val_{j}.pt')
                        torch.save(data, path)
                        j+=1
                elif(self.split == 'train'):
                    j=0
                    for i in tqdm(self.train_indices):
                        node_features, coo, edge_features, pos = graph_utils.RAG(self.data[i][0], n_nodes=self.nodes)
                        data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                    edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                    edge_attr=torch.from_numpy(edge_features),
                                    y=self.data[i][1],
                                    pos=torch.from_numpy(pos))
                        path = os.path.join('data/processed/', f'RAG_train_{j}.pt')
                        torch.save(data, path)
                        j+=1
        
    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        if(self.split == 'train'):
            return self.train_indices.shape[0]
        elif(self.split == 'val'):
            return self.val_indices.shape[0]
        else:
            return len(self.data)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if(not self.hierarchy):
            if(self.multi_RAG):
                if(self.split == 'train'):
                    data = torch.load(os.path.join(self.processed_dir, f'MULTI_RAG_train_{idx}.pt'))
                elif(self.split == 'val'):
                    data = torch.load(os.path.join(self.processed_dir, f'MULTI_RAG_val_{idx}.pt'))
                else:
                    data = torch.load(os.path.join(self.processed_dir, f'MULTI_RAG_test_{idx}.pt'))
            else:
                if(self.split == 'train'):
                    data = torch.load(os.path.join(self.processed_dir, f'RAG_train_{idx}.pt'))
                elif(self.split == 'val'):
                    data = torch.load(os.path.join(self.processed_dir, f'RAG_val_{idx}.pt'))
                else:
                    data = torch.load(os.path.join(self.processed_dir, f'RAG_test_{idx}.pt'))
        else:
            if(self.complete and self.knn):
                load_str = 'complete_hierarchy'
            elif(self.knn):   
                load_str = 'knn_hierarchy'
            else:
                if(self.cuts and self.rag): 
                    load_str = 'hierarchy+RAG+CUTS'
                elif(self.cuts):
                    load_str = 'hierarchy+CUTS'
                elif(self.rag):
                    load_str = 'hierarchy+RAG'
                else:
                    load_str = 'hierarchy'
            if(self.split == 'train'):
               data = torch.load(os.path.join(self.processed_dir, load_str+f'_train_{idx}.pt'))
            elif(self.split == 'val'):
                data = torch.load(os.path.join(self.processed_dir, load_str+f'_val_{idx}.pt'))
            else:
                data = torch.load(os.path.join(self.processed_dir, load_str+f'_test_{idx}.pt'))
                    
        return data

class MultiGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'reduced_index':
            return 1
        if key == 'graph_index':
            return self.ng
        return super().__inc__(key, value, *args, **kwargs)
    
class MG_Cifar10_graphs(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, split='train', nodes=100, hierarchy=True, canonized=False):
        self.split = split
        self.nodes = nodes
        self.hierarchy = hierarchy 
        self.canonized = canonized

        if(self.split == 'train'):
            self.train_indices=np.load("train_indices-cifar.npy")
            self.data = CIFAR10("data/raw/", download=True, transform=graph_utils.be_np, 
                target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        elif(self.split == 'val'):
            self.val_indices=np.load("val_indices-cifar.npy")
            self.data = CIFAR10("data/raw/", download=True, transform=graph_utils.be_np, 
                target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))   
        else:
            self.data = CIFAR10("data/raw/", download=True, transform=graph_utils.be_np, train=False,
                target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

        super(MG_Cifar10_graphs, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return 'cifar-10-python.tar.gz'

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        print("Using Superpixel MULTI GRAPHS w/ Hierarchy!")
        load_str = 'MULTI_GRAPH_HIERARCHY'
        if(self.split == 'train'):
            return [load_str+f'_train_{i}.pt' for i in range(self.train_indices.shape[0])]
        elif(self.split == 'val'):
            return [load_str+f'_val_{i}.pt' for i in range(self.val_indices.shape[0])]
        else:
            return [load_str+f'_test_{i}.pt' for i in range(len(self.data))]
            
                  
    def download(self):
        pass

    def process(self):
        
        print("Using superpixel MULTI GRAPH w/ Hierarchy target "+str(self.nodes)+" number of nodes") 
        if(self.split == 'test'):
            for i in tqdm(range(len(self.data))):
                #print(i)
                node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_superpixel_hierarchy(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                data = MultiGraphData(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                            edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                            edge_attr=torch.from_numpy(edge_features),
                            y=self.data[i][1],
                            pos=torch.from_numpy(pos),
                            graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                            reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                            ng=torch.from_numpy(ng))
                path = os.path.join('data/processed/', f'MULTI_GRAPH_HIERARCHY_test_{i}.pt')
                torch.save(data, path)
        elif(self.split == 'val'):
            j=0
            for i in tqdm(self.val_indices):
                #print(i)
                node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_superpixel_hierarchy(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                data = MultiGraphData(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                            edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                            edge_attr=torch.from_numpy(edge_features),
                            y=self.data[i][1],
                            pos=torch.from_numpy(pos),
                            graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                            reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                            ng=torch.from_numpy(ng))
                path = os.path.join('data/processed/', f'MULTI_GRAPH_HIERARCHY_val_{j}.pt')
                torch.save(data, path)
                j+=1
        elif(self.split == 'train'):
            j=0
            for i in tqdm(self.train_indices):
                node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_superpixel_hierarchy(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                data = MultiGraphData(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                            edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                            edge_attr=torch.from_numpy(edge_features),
                            y=self.data[i][1],
                            pos=torch.from_numpy(pos),
                            graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                            reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                            ng=torch.from_numpy(ng))
                path = os.path.join('data/processed/', f'MULTI_GRAPH_HIERARCHY_train_{j}.pt')
                torch.save(data, path)
                j+=1
        
    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        if(self.split == 'train'):
            return self.train_indices.shape[0]
        elif(self.split == 'val'):
            return self.val_indices.shape[0]
        else:
            return len(self.data)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        load_str = 'MULTI_GRAPH_HIERARCHY'
        if(self.split == 'train'):
            data = torch.load(os.path.join(self.processed_dir, load_str+f'_train_{idx}.pt'))
        elif(self.split == 'val'):
            data = torch.load(os.path.join(self.processed_dir, load_str+f'_val_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, load_str+f'_test_{idx}.pt'))
                    
        return data
    
    
class MG_STL10_graphs(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, split='train', nodes=100, hierarchy=True, canonized=False, only_hierarchy=False, rag_ones=False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.split = split
        self.nodes = nodes
        self.hierarchy = hierarchy 
        self.canonized = canonized
        self.rag_ones = rag_ones
        self.only_hierarchy = only_hierarchy
        if(self.split == 'train'):
            self.train_indices=np.load("train_indices-stl.npy")
            
            self.data = STL10("data/raw/", download=True, transform=graph_utils.be_np,split='train',
                        target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        elif(self.split == 'val'):
            self.val_indices=np.load("val_indices-stl.npy")

            self.data = STL10("data/raw/", download=True, transform=graph_utils.be_np, split='train', 
                        target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))   
        else:
            self.data = STL10("data/raw", download=True, transform=graph_utils.be_np, split='test',
                        target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

        super(MG_STL10_graphs, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return 'stl10_binary.tar.gz'

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        if(not self.hierarchy):
            print("Using Superpixel MULTI GRAPHS RAG!")
            if(self.split == 'train'):
                return [f'MULTI_GRAPHS_RAG_train_{i}.pt' for i in range(self.train_indices.shape[0])]
            elif(self.split == 'val'):
                return [f'MULTI_GRAPHS_RAG_val_{i}.pt' for i in range(self.val_indices.shape[0])]
            else:
                return [f'MULTI_GRAPHS_RAG_test_{i}.pt' for i in range(len(self.data))]
        elif(self.only_hierarchy):
            print("Using Superpixel MULTI GRAPHS **ONLY** Hierarchy!")
            load_str = 'MULTI_GRAPH_ONLY_HIERARCHY'
            if(self.split == 'train'):
                return [load_str+f'_train_{i}.pt' for i in range(self.train_indices.shape[0])]
            elif(self.split == 'val'):
                return [load_str+f'_val_{i}.pt' for i in range(self.val_indices.shape[0])]
            else:
                return [load_str+f'_test_{i}.pt' for i in range(len(self.data))]
        else:
            print("Using Superpixel MULTI GRAPHS w/ Hierarchy!")
            load_str = 'MULTI_GRAPH_HIERARCHY'
            if(self.split == 'train'):
                return [load_str+f'_train_{i}.pt' for i in range(self.train_indices.shape[0])]
            elif(self.split == 'val'):
                return [load_str+f'_val_{i}.pt' for i in range(self.val_indices.shape[0])]
            else:
                return [load_str+f'_test_{i}.pt' for i in range(len(self.data))]
            
                  
    def download(self):
        pass

    def process(self):
        if(not self.hierarchy):
            print("Using superpixel MULTI RAG target "+str(self.nodes)+" number of nodes") 
            if(self.split == 'test'):
                for i in tqdm(range(len(self.data))):
                    #print(i)
                    node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_multi_RAG(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                    data = MultiGraphData(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos),
                                graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                                reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                                ng=torch.from_numpy(ng))
                    path = os.path.join('data/processed/', f'MULTI_GRAPHS_RAG_test_{i}.pt')
                    torch.save(data, path)
            elif(self.split == 'val'):
                j=0
                for i in tqdm(self.val_indices):
                    #print(i)
                    node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_multi_RAG(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                    data = MultiGraphData(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos),
                                graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                                reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                                ng=torch.from_numpy(ng))
                    path = os.path.join('data/processed/', f'MULTI_GRAPHS_RAG_val_{j}.pt')
                    torch.save(data, path)
                    j+=1
            elif(self.split == 'train'):
                j=0
                for i in tqdm(self.train_indices):
                    node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_multi_RAG(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                    data = MultiGraphData(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos),
                                graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                                reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                                ng=torch.from_numpy(ng))
                    path = os.path.join('data/processed/', f'MULTI_GRAPHS_RAG_train_{j}.pt')
                    torch.save(data, path)
                    j+=1
        elif(self.only_hierarchy):
            print("Using superpixel MULTI GRAPH **ONLY** Hierarchy target "+str(self.nodes)+" number of nodes") 
            if(self.split == 'test'):
                for i in tqdm(range(len(self.data))):
                    #print(i)
                    node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_hierarchy(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                    data = MultiGraphData(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos),
                                graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                                reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                                ng=torch.from_numpy(ng))
                    path = os.path.join('data/processed/', f'MULTI_GRAPH_ONLY_HIERARCHY_test_{i}.pt')
                    torch.save(data, path)
            elif(self.split == 'val'):
                j=0
                for i in tqdm(self.val_indices):
                    #print(i)
                    node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_hierarchy(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                    data = MultiGraphData(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos),
                                graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                                reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                                ng=torch.from_numpy(ng))
                    path = os.path.join('data/processed/', f'MULTI_GRAPH_ONLY_HIERARCHY_val_{j}.pt')
                    torch.save(data, path)
                    j+=1
            elif(self.split == 'train'):
                j=0
                for i in tqdm(self.train_indices):
                    node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_hierarchy(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized)
                    data = MultiGraphData(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos),
                                graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                                reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                                ng=torch.from_numpy(ng))
                    path = os.path.join('data/processed/', f'MULTI_GRAPH_ONLY_HIERARCHY_train_{j}.pt')
                    torch.save(data, path)
                    j+=1
        else:
            print("Using superpixel MULTI GRAPH w/ Hierarchy target "+str(self.nodes)+" number of nodes") 
            if(self.rag_ones):
                print("Rag edge features = ONE")
            if(self.split == 'test'):
                for i in tqdm(range(len(self.data))):
                    #print(i)
                    node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_superpixel_hierarchy(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized, rag_ones=self.rag_ones)
                    data = MultiGraphData(x=torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos),
                                graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                                reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                                ng=torch.from_numpy(ng))
                    path = os.path.join('data/processed/', f'MULTI_GRAPH_HIERARCHY_test_{i}.pt')
                    torch.save(data, path)
            elif(self.split == 'val'):
                j=0
                for i in tqdm(self.val_indices):
                    #print(i)
                    node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_superpixel_hierarchy(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized, rag_ones=self.rag_ones)
                    data = MultiGraphData(x=torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos),
                                graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                                reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                                ng=torch.from_numpy(ng))
                    path = os.path.join('data/processed/', f'MULTI_GRAPH_HIERARCHY_val_{j}.pt')
                    torch.save(data, path)
                    j+=1
            elif(self.split == 'train'):
                j=0
                for i in tqdm(self.train_indices):
                    node_features, coo, edge_features, pos, graph_index, reduced_index, ng = graph_utils.MG_superpixel_hierarchy(self.data[i][0], n_nodes=self.nodes, canonized=self.canonized, rag_ones=self.rag_ones)
                    data = MultiGraphData(x=torch.from_numpy(node_features).type(torch.FloatTensor),
                                edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                                edge_attr=torch.from_numpy(edge_features),
                                y=self.data[i][1],
                                pos=torch.from_numpy(pos),
                                graph_index=torch.from_numpy(graph_index).type(torch.LongTensor), 
                                reduced_index=torch.from_numpy(reduced_index).type(torch.LongTensor), 
                                ng=torch.from_numpy(ng))
                    path = os.path.join('data/processed/', f'MULTI_GRAPH_HIERARCHY_train_{j}.pt')
                    torch.save(data, path)
                    j+=1
        
    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        if(self.split == 'train'):
            return self.train_indices.shape[0]
        elif(self.split == 'val'):
            return self.val_indices.shape[0]
        else:
            return len(self.data)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if(not self.hierarchy):
            if(self.split == 'train'):
                data = torch.load(os.path.join(self.processed_dir, f'MULTI_GRAPHS_RAG_train_{idx}.pt'))
            elif(self.split == 'val'):
                data = torch.load(os.path.join(self.processed_dir, f'MULTI_GRAPHS_RAG_val_{idx}.pt'))
            else:
                data = torch.load(os.path.join(self.processed_dir, f'MULTI_GRAPHS_RAG_test_{idx}.pt'))
        elif(self.only_hierarchy):
            load_str = 'MULTI_GRAPH_ONLY_HIERARCHY'
            if(self.split == 'train'):
               data = torch.load(os.path.join(self.processed_dir, load_str+f'_train_{idx}.pt'))
            elif(self.split == 'val'):
                data = torch.load(os.path.join(self.processed_dir, load_str+f'_val_{idx}.pt'))
            else:
                data = torch.load(os.path.join(self.processed_dir, load_str+f'_test_{idx}.pt'))
        else:
            load_str = 'MULTI_GRAPH_HIERARCHY'
            if(self.split == 'train'):
               data = torch.load(os.path.join(self.processed_dir, load_str+f'_train_{idx}.pt'))
            elif(self.split == 'val'):
                data = torch.load(os.path.join(self.processed_dir, load_str+f'_val_{idx}.pt'))
            else:
                data = torch.load(os.path.join(self.processed_dir, load_str+f'_test_{idx}.pt'))
                    
        return data