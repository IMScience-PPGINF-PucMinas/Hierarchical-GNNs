HIERARCHICAL GRAPH NEURAL NETWORKS BASED ON MULTI-SCALE IMAGE REPRESENTATIONS
=====
This is the MsC Dissertation project of [João Pedro Oliveira Batisteli](https://lattes.cnpq.br/8128547685252443) for evaluating the impact of hierarchical segmentation for create graph in image-graph classification.

##Abstract

Image representation as graphs can enhance the understanding of image semantics and facilitate multi-scale image representation. However, existing methods often overlook the significance of the relationship between elements at each scale or fail to encode the hierarchical relationship between graph elements. Moreover, the performance of GNNs still lags behind traditional CNNs due to the loss of information during dimensionality reduction when creating graphs from images. To cope with that, we introduce four novel approaches for graph construction from images. These approaches utilize a hierarchical image segmentation technique to generate segmentation at multiple scales and, in one of them, incorporate edges to encode the relationships at each scale.

## Getting started

### Prerequisites

0. Clone this repository

```
git clone "https://github.com/IMScience-PPGINF-PucMinas/Hierarchical-GNNs.git"
cd "Hierarchical-GNNs"
```

1. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate hierarchical_gnns
```

2. Prepare feature files

The program will download the CIFAR-10 and STL-10 datasets and generate the graphs at first time you execute the training or test script. The graphs will be saved in the `data/processed` for future use.

If you want to change the target number of superpixels, you need to modify the `nodes` parameter in the training files for each model. The nodes parameter determines how many superpixels the model will generate. For the HIGSI model, the training file is MG_train.py. You can find the nodes parameter on lines 97, 98 and 99. For the BRM model, the training file is train.py. You can find the nodes parameter on lines 86, 87 and 88.

The dataset will be saved in the `data/processed` folder after you create it. If you want to generate new graphs, you have to delete the existing files in that folder first. Otherwise, the program will use the old files and not create new ones.

### Training and Inference

1. Before you start the model training, you need to launch the mlflow server in a terminal. This will allow you to monitor the model metrics during the training process. To launch the mlflow server, use the following command:

```bash
mlflow ui
```

To start the training scripts, you need to open another terminal and specify the random seed you want to use. The random seed will affect the initialization of the model parameters and the shuffling of the data. You can use any integer value for the random seed.

To run the training script with a given random seed for the BRM model with seed 1, for example, use the following command:

```bash
python train.py 1
```

For the HIGSI model:

```bash
python RG_train.py 1
```

If you want to reproduce the results from the article, you need to run different shell scripts for each model. The shell scripts will run the experiments with the same random seeds that were used in the article. For the HIGHSI model, run the `MG_train_runs.sh` script. For the BRM model, run the `train_runs.sh` script.

2. To perform model testing, it is only necessary to pass the seed that the model was trained on.
For the BRM model with seed 1, for example, run the command:

```bash
python test.py 1
```

For the HIGSI model:

```bash
python RG_test.py 1
```

Don't forget to change directory to the model weights you want to test.

## Citations

If you find this code useful for your research, consider cite our papers:

```
@INPROCEEDINGS{232350,
    AUTHOR="João Pedro Oliveira Batisteli and Zenilton Kleber Patrocínio Jr and Silvio Guimarães",
    TITLE="Hierarchical Graph Convolutional Networks for Image Classification",
    BOOKTITLE="BRACIS 2023",
    ADDRESS="Belo Horizonte, MG",
    DAYS="25-29",
    MONTH="sep",
    YEAR="2023",
}
```

@inproceedings{batisteli2023multiscale,
  title={Multi-scale image graph representation: a novel GNN approach for image classification through scale importance estimation},
  author={Batisteli, Jo{\~a}o Pedro Oliveira and Guimar{\~a}es, Silvio Jamil Ferzoli and do Patroc{\'\i}nio J{\'u}nior, Zenilton Kleber Gon{\c{c}}alves},
  booktitle={IEEE International Symposium on Multimedia (ISM)},
  year={2023}
}

## Contact

João Pedro Oliveira Batisteli: <joao.batisteli@sga.pucminas.br>
