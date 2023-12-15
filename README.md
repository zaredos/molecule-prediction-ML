# molecule-prediction-ML
A Machine Learning model which uses both graph nodes and regular torch tensors to predict molecular properties such as Activation or Inhibition, Toxicity Prediction, Binding Affinity, Dose-Response Relationships. Final project for COSI 149B.

## Project 3: Molecule Toxicity Prediction using Hybrid Model
Molecules can be represented as 2D graphs, with atoms as nodes and chemical bonds as edges. In the Hybrid Model we developed, we combined the given graph nodes as well as new non-graph nodes to see insights that might be missed out when only doing one kind of data. But before that, we are going to test to see what descriptors will make the model have the highest accuracy, and that’s why we created the first ipynb: rand_descriptors. More descriptions can be found in the ipynb as comments.
From these 33 descriptors, we created a looping function that randomly selects 10 of them to be used in a GCN model and compares the validation accuracy to see which descriptor combination is the most beneficial for the results.

```
descriptor_list = [desc.FpDensityMorgan1,
                  desc.FpDensityMorgan2,
                  desc.FpDensityMorgan3,
                  desc.BalabanJ,
                  desc.BertzCT,
                  desc.Ipc,
                  desc.Kappa1,
                  desc.Kappa2,
                  desc.Kappa3,
                  desc.MolWt,
                  desc.MolLogP,
                  desc.NumRotatableBonds,
                  desc.NumHAcceptors,
                  desc.NumHDonors,de
                  desc.NumHeteroatoms,
                  desc.NumValenceElectrons,
                  desc.MolMR,
                  desc.NumRadicalElectrons,
                  desc.RingCount,
                  desc.NumAromaticRings,
                  desc.NumAliphaticRings,
                  desc.NumSaturatedRings,
                  desc.NumAromaticHeterocycles,
                  desc.NumAromaticCarbocycles,
                  desc.NumSaturatedHeterocycles,
                  desc.NumSaturatedCarbocycles,
                  desc.NumAliphaticHeterocycles,
                  desc.NumAliphaticCarbocycles,
                  desc.NOCount,
                  desc.NHOHCount,
                  desc.FractionCSP3,
                  desc.LabuteASA,
                  desc.TPSA,
]
```

After that, we will be using hybrid_model.ipynb. We will first import the datasets:

```
import torch
import torch_geometric
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
import pickle

# Load datasets. The training and validation sets contain both molecules and their property labels. The test set only contain molecules.
# There are 12 property tasks for prediction. Some properties labels are missing (i.e., nan). You can ignore them.

train_dataset = torch.load("train_data.pt")
valid_dataset = torch.load("valid_data.pt")
test_dataset = torch.load("test_data.pt")
print(f'Size of training set: {len(train_dataset)}')
print(f'Size of validation set: {len(valid_dataset)}')
print(f'Size of test set: {len(test_dataset)}')
```

And then we build the Hybrid Model
## Build Hybrid Model
This is our take on integrating Graph-Based and Non-Graph Features.
The HybridModel class in this implementation is a neural network model designed to handle both graph-structured data (from the individual atoms and their edge features) and non-graph features (chemical descriptors on the molecular level). This model is particularly useful in scenarios where additional features (that are not part of the graph structure) can complement the graph data for more accurate predictions.
Components of the Hybrid Model Graph-Based Classifier

- Atom Encoding: Uses the given AtomEncoder to transform node features into a higher-dimensional space using embedding.
- GCNs: Consists of three GCNConv layers to process the graph data. Each GCNConv layer transforms the node features while considering the graph structure.
- Linear Transformation: We applied a linear layer (lin_graph) to the graph-level features after pooling.
Non-Graph Classifier
- Non-Graph Feature Encoding: Uses a modified AtomEncoder without embedding ModifiedAtomEncoder (created and tested in rand_descriptors.ipynb) to transform non-graph features, which allows for float values and values greater than 100 (which the original AtomEncoder does not allow).
- Fully Connected Layers: Two fully connected (fc) layers process the non-graph features.
Combined Architecture
- Integration of Features: The model combines the features extracted from both graph-based and non-graph components.
- Final Classification Layers: After concatenation, additional fc layers are used to make final predictions based on combined features.
Forward Pass
- Graph Component Handling: Processes graph features (x) and uses global mean pooling (gap) to obtain a fixed-size representation for each graph in the batch.
- Non-Graph Component Handling: Processes non-graph features (y) in a similar manner.
- Combination and Prediction: Concatenates the outputs from both components and passes them through final layers to produce predictions.
Below is our implementation:

```
 def reshape_tensor(tensor):
   reshaped_tensor = tensor.view(16, 32)
   return reshaped_tensor

  # The given AtomEncoder
class AtomEncoder(torch.nn.Module):
   def __init__(self, hidden_channels):
       super(AtomEncoder, self).__init__()
       self.embeddings = torch.nn.ModuleList()
       for i in range(9):
           self.embeddings.append(torch.nn.Embedding(100, hidden_channels))
   def reset_parameters(self):
       for embedding in self.embeddings:
           embedding.reset_parameters()
   def forward(self, x):
       if x.dim() == 1:
           x = x.unsqueeze(1)
       out = 0
       for i in range(x.size(1)):
           out += self.embeddings[i](x[:, i])
return out

# Our Encoder (does not use the edge_index or embeddings, can get about 65% accuracy)
class ModifiedAtomEncoder(torch.nn.Module):
   def __init__(self, hidden_channels, num_node_features):
       super(ModifiedAtomEncoder, self).__init__()

         self.linear_layers = torch.nn.ModuleList()
       for i in range(num_node_features):
           self.linear_layers.append(torch.nn.Linear(1, hidden_channels))
   def reset_parameters(self):
       for linear in self.linear_layers:
           torch.nn.init.xavier_uniform_(linear.weight)
           torch.nn.init.zeros_(linear.bias)
   def forward(self, x):
       if x.dim() == 1:
           x = x.unsqueeze(1)
       out = 0
       for i in range(x.size(1)):
           out += self.linear_layers[i](x[:, i:i+1])
       return out

class GCN_noGraph(torch.nn.Module):
   def __init__(self, hidden_channels, num_node_features, num_classes):
       super(GCN_noGraph, self).__init__()

         torch.manual_seed(42)
       self.emb = AtomEncoder(hidden_channels=32, num_node_features=num_node_features)
       self.lin = Linear(hidden_channels, num_classes)
   def forward(self, batch):
       x, batch_size = batch.descriptors, batch.batch
       x = self.emb(x)
       # 2. Readout layer
       x = gap(x, batch_size)  # [batch_size, hidden_channels]
       # 3. Apply a final classifier
       x = F.dropout(x, p=0.5, training=self.training)
       x = self.lin(x)
       return x

 # The given model (uses the edge_index, can get about 74% accuracy)
class GCN(torch.nn.Module):
   def __init__(self, hidden_channels, num_node_features, num_classes):
       super(GCN, self).__init__()
       torch.manual_seed(42)
       self.emb = AtomEncoder(hidden_channels=32)
       self.conv1 = GCNConv(hidden_channels, hidden_channels)
       self.conv2 = GCNConv(hidden_channels, hidden_channels)

         self.conv3 = GCNConv(hidden_channels, hidden_channels)
       self.lin = Linear(hidden_channels, num_classes)
   def forward(self, batch):
       x, edge_index, batch_size = batch.x, batch.edge_index, batch.batch
       x = self.emb(x)
       # 1. Obtain node embeddings
       x = self.conv1(x, edge_index)
       x = x.relu()
       x = self.conv2(x, edge_index)
       x = x.relu()
       x = self.conv3(x, edge_index)
       # 2. Readout layer
       x = gap(x, batch_size)  # [batch_size, hidden_channels]
       # 3. Apply a final classifier
       x = F.dropout(x, p=0.5, training=self.training)
       x = self.lin(x)
       return x

 # Hybrid model (uses both the edge_index and the non-graph features)
class HybridModel(torch.nn.Module):

     def __init__(self, hidden_channels, num_node_features, num_classes,
num_features_nongraph):
       super(HybridModel, self).__init__()
       # Graph-based Classifier
       self.emb = AtomEncoder(hidden_channels=hidden_channels)
       self.non_graph_emb = ModifiedAtomEncoder(hidden_channels=hidden_channels,
num_node_features=num_features_nongraph)
       self.conv1 = GCNConv(hidden_channels, hidden_channels)
       self.conv2 = GCNConv(hidden_channels, hidden_channels)
       self.conv3 = GCNConv(hidden_channels, hidden_channels)
       self.lin_graph = Linear(hidden_channels, hidden_channels)
       # Non-graph Classifier
       self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
       self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
       # Combined layers
       self.fc3 = torch.nn.Linear(2 * hidden_channels, hidden_channels)  # Combining
graph and non-graph outputs
       self.fc4 = torch.nn.Linear(hidden_channels, num_classes)
   def forward(self, batch):
       # Graph component (x is the graph features, y is the non-graph features)
       x, edge_index, batch_index, y = batch.x, batch.edge_index, batch.batch,
batch.descriptors

  x = self.emb(x)
x = F.relu(self.conv1(x, edge_index))
x = F.relu(self.conv2(x, edge_index))
x = F.relu(self.conv3(x, edge_index))
x = gap(x, batch_index)  # [batch_size, hidden_channels]
x = F.dropout(x, p=0.5, training=self.training)
x = self.lin_graph(x)
x = x.float() # Convert to float for concatenation
# Non-graph component
y = self.non_graph_emb(y)
y = F.relu(self.fc1(y))
y = gap(y, batch_index)  # [batch_size, hidden_channels]
# 3. Apply a final classifier
y = F.dropout(y, p=0.5, training=self.training)
y = self.fc2(y)
# Combine (via simple concatenation)
z = torch.cat([x, y], dim=1)
z = F.relu(self.fc3(z))
z = self.fc4(z)
return z
```

## Feature Engineering for Graph Datasets Process Overview
1. Combining Datasets: All datasets are combined into a single list named datasets for streamlined processing.
1. Iterating Over Each Graph:
- The code iterates through each dataset in datasets, and then through each graph within these datasets.
1. Molecular Descriptor Calculation:
- For every graph, the corresponding molecular structure is retrieved using Chem.MolFromSmiles.
- A series of molecular descriptors are computed for each molecule. These descriptors include:
  - FpDensityMorgan1,FpDensityMorgan2,FpDensityMorgan3: Morgan fingerprint densities.
  - HeavyAtomMolWt:Weightofheavyatomsinthemolecule.
  - NumHAcceptors:Numberofhydrogenbondacceptors.
  - NHOHCount:Countofnitrogenandhydroxidegroups.
  - Kappa3:Ashapedescriptor.
  - NOCount:Countofnitrogenandoxygenatoms.
  - HallKierAlpha:Hall-KierAlphavalue.
  - MinEStateIndex:Minimumelectrotopologicalstateindex.
  - MolWt:Molecularweight.
  - BalabanJ:BalabanJvalue.
  - MolLogP:Octanol-waterpartitioncoefficient.
  - PEOE_VSA6,SlogP_VSA2,SMR_VSA7:VariousVSAdescriptors.
1. Tensor Conversion and Assignment:
- The computed descriptors are converted into a PyTorch tensor with torch.float32 data type.
- These new features are then assigned to the descriptors attribute of each graph for later use in models.
  
## Model init Model creation

Here we created a Hybrid Model with 32 hidden channels, the number of node features, 12 output classes, and the number of non-graph features.
- num_node_features:Thenumberofnodefeaturesinthegraph,determinedfromthe shape of the x attribute of the first graph in the training dataset.
- num_features_nongraph:Thenumberofnon-graphfeatures,determinedfromthe shape of the descriptors attribute of the first graph in the training dataset.
Although both the regular GCN and nongraph GCN classifiers have worked perfectly individually, when combined in HybridModel, we have issues with matching the tensor sizes for concatenating the two computed tensors together.
- Dimension 0 changes slightly each time the code runs, and we are frustrated as to why it is not consistent.
Loss function and Optimizer
We used an Adam optimizer with a learning rate of 0.001 for its efficiency in handling sparse gradients and adaptive learning rates. For the loss function, we chose Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss). This is appropriate for binary classification.
Dataloader setup
- Batch Size: A batch size of 32 is chosen for training, which balances the computational efficiency and memory usage.
- DataLoaders for Datasets: Separate DataLoader instances are created for the training, validation, and test datasets. These loaders handle the batching and shuffling of data, ensuring efficient and randomized access to the datasets during training and evaluation.
- Shuffling: The training data loader shuffles the data to ensure randomness in batch selection, which helps in generalizing the model. The validation and test loaders do not shuffle the data, as the order does not impact model evaluation.
