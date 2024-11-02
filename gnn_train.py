import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import json
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wget
import zipfile

class WebQSPDataset(Dataset):
    """
    WebQuestionsSP dataset loader with consistent tensor dimensions
    """
    def __init__(self, root: str, split: str = 'train', feature_dim: int = 1538):
        super().__init__(root)
        self.split = split
        self.data_path = os.path.join(root, "WebQSP")
        self.feature_dim = feature_dim
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load and process WebQSP data"""
        file_path = os.path.join(
            self.data_path, 
            f'WebQSP.{self.split}.json'
        )
        
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
            
        processed_data = []
        for item in raw_data['Questions']:
            if not item['Parses']:
                continue
                
            parse = item['Parses'][0]
            
            # Extract answers
            answers = []
            if 'Answers' in parse:
                answers = [ans['AnswerArgument'] for ans in parse['Answers']]
            
            # Extract entities and relations
            entities = []
            relations = []
            
            if 'TopicEntityMid' in parse:
                entities.append(parse['TopicEntityMid'])
            
            entities.extend(answers)
            
            if 'InferentialChain' in parse:
                relations = parse['InferentialChain']
            
            # Ensure we have at least one entity
            if not entities:
                continue
                
            # If no relations, create a dummy self-relation
            if not relations:
                relations = ['self_loop']
            
            # Remove duplicates while preserving order
            entities = list(dict.fromkeys(entities))
            relations = list(dict.fromkeys(relations))
            
            processed_item = {
                'question': item['ProcessedQuestion'],
                'answers': answers,
                'entities': entities,
                'relations': relations,
                'id': item['QuestionId']
            }
            processed_data.append(processed_item)
            
        return processed_data
    
    def len(self) -> int:
        return len(self.data)
        
    def get(self, idx: int) -> Data:
        item = self.data[idx]
        
        # Create mappings
        entity_to_idx = {entity: idx for idx, entity in enumerate(item['entities'])}
        relation_to_idx = {rel: idx for idx, rel in enumerate(item['relations'])}
        
        num_entities = len(entity_to_idx)
        num_relations = len(relation_to_idx)
        
        # Create node features with fixed dimension
        x = torch.zeros((num_entities, self.feature_dim))
        torch.nn.init.normal_(x)
        
        # Create edges
        edge_index = []
        edge_attr = []
        
        # Create fully connected graph with relation types
        for rel in item['relations']:
            rel_idx = relation_to_idx[rel]
            for i in range(num_entities):
                for j in range(num_entities):
                    if i != j or rel == 'self_loop':
                        edge_index.append([i, j])
                        edge_attr.append(rel_idx)
        
        # Convert to tensors
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        else:
            edge_index = torch.arange(num_entities).repeat(2, 1)
            edge_attr = torch.zeros(num_entities, dtype=torch.long)
        
        # Create target - reshape to match model output
        y = torch.zeros(num_entities, 1)  # Changed to 2D tensor
        for ans in item['answers']:
            if ans in entity_to_idx:
                y[entity_to_idx[ans]][0] = 1
                
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_entities,
            batch=None
        )

    @property
    def num_node_features(self) -> int:
        return self.feature_dim
    
    @property
    def num_classes(self) -> int:
        return 1  # Binary classification per node

class GNNModel(torch.nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Add batch normalization layers
        self.input_bn = torch.nn.BatchNorm1d(input_dim)
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        # Apply input batch normalization
        x = self.input_bn(x)
        x = self.input_layer(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i > 0:  # Skip connection
                x = x + identity
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.output_layer(x)
        return x

def calculate_class_weights(dataset: Dataset) -> torch.Tensor:
    """Calculate class weights based on class distribution"""
    pos_count = 0
    total_count = 0
    
    for data in dataset:
        pos_count += data.y.sum().item()
        total_count += len(data.y)
    
    neg_count = total_count - pos_count
    pos_weight = neg_count / (pos_count + 1e-10)  # Add small epsilon to prevent division by zero
    return torch.tensor([pos_weight])

def find_optimal_threshold(val_loader: DataLoader,
                         model: GNNModel,
                         device: torch.device) -> float:
    """Find optimal classification threshold using validation set"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            probs = torch.sigmoid(out)
            all_probs.append(probs.cpu())
            all_labels.append(batch.y.cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Try different thresholds
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs > threshold).astype(float)
        # Calculate F1 score
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold

def train_model(model: GNNModel,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int,
                learning_rate: float,
                device: torch.device,
                save_dir: str):
    """Train GNN model with improved class balance handling"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate class weights from training data
    pos_weight = calculate_class_weights(train_loader.dataset).to(device)
    
    # Use AdamW optimizer for better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # More aggressive learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6, verbose=True,
        min_lr=1e-6, threshold=1e-4
    )
    
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    
    best_val_f1 = 0
    best_epoch = 0
    patience = 40  # Increased patience for better convergence
    no_improve = 0
    best_threshold = 0.5
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_predictions = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index)
            
            # Use weighted BCE loss
            loss = F.binary_cross_entropy_with_logits(
                out, batch.y.float(), 
                pos_weight=pos_weight
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions using current best threshold
            probs = torch.sigmoid(out)
            preds = (probs > best_threshold).float()
            train_predictions.append(preds.cpu())
            train_labels.append(batch.y.cpu())
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                
                val_batch_loss = F.binary_cross_entropy_with_logits(
                    out, batch.y.float(),
                    pos_weight=pos_weight
                )
                val_loss += val_batch_loss.item()
                
                probs = torch.sigmoid(out)
                val_predictions.append(probs.cpu())
                val_labels.append(batch.y.cpu())
        
        # Find optimal threshold on validation set
        val_predictions = torch.cat(val_predictions)
        val_labels = torch.cat(val_labels)
        
        # Update best threshold
        best_threshold = find_optimal_threshold(val_loader, model, device)
        
        # Calculate metrics using best threshold
        val_preds = (val_predictions > best_threshold).float()
        val_metrics = calculate_metrics(val_preds, val_labels)
        
        train_predictions = torch.cat(train_predictions)
        train_labels = torch.cat(train_labels)
        train_metrics = calculate_metrics(train_predictions, train_labels)
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Logging
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Threshold/best', best_threshold, epoch)
        for metric, value in train_metrics.items():
            writer.add_scalar(f'{metric}/train', value, epoch)
        for metric, value in val_metrics.items():
            writer.add_scalar(f'{metric}/val', value, epoch)
        
        # Learning rate scheduling based on validation F1
        scheduler.step(val_metrics['f1'])
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'best_threshold': best_threshold,
                'val_metrics': val_metrics,
            }, os.path.join(save_dir, 'best_model.pt'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
        
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'  Best Threshold: {best_threshold:.4f}')
        print('  Train Metrics:', {k: f'{v:.4f}' for k, v in train_metrics.items()})
        print('  Val Metrics:', {k: f'{v:.4f}' for k, v in val_metrics.items()})
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
    
    writer.close()
    return best_epoch, best_val_f1, best_threshold

def calculate_metrics(predictions: torch.Tensor, 
                     labels: torch.Tensor) -> Dict[str, float]:
    """Calculate evaluation metrics with class balance considerations"""
    predictions = predictions.squeeze()
    labels = labels.squeeze()
    
    tp = (predictions * labels).sum().float()
    fp = (predictions * (1 - labels)).sum().float()
    fn = ((1 - predictions) * labels).sum().float()
    tn = ((1 - predictions) * (1 - labels)).sum().float()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    
    # Add class-specific metrics
    specificity = tn / (tn + fp + 1e-10)
    balanced_accuracy = (recall + specificity) / 2
    
    # Add positive and negative predictive values
    ppv = precision  # Same as precision
    npv = tn / (tn + fn + 1e-10)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'accuracy': accuracy.item(),
        'balanced_acc': balanced_accuracy.item(),
        'specificity': specificity.item(),
        'ppv': ppv.item(),
        'npv': npv.item()
    }
    
def gnn_inference(model, graph_data):
    """
    Perform inference on a GNN model for node classification.

    Args:
        model (torch.nn.Module): Trained GNN model.
        graph_data (torch_geometric.data.Data): Graph data containing node features, edge index, etc.

    Returns:
        torch.Tensor: Predicted class labels for each node in the graph.
    """
    model.eval()  # Set model to evaluation mode

    # Ensure the input is in the correct device (CPU/GPU)
    graph_data = graph_data.to(model.device)
    
    # Perform forward pass through the model
    with torch.no_grad():
        output = model(graph_data.x, graph_data.edge_index)

    # Apply softmax to get probabilities (if needed) or directly take the argmax for class labels
    predictions = output.argmax(dim=1)

    return predictions

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset paths
    data_root = ''
    save_dir = 'model_checkpoints'
    
    # Load dataset
    dataset = WebQSPDataset(root=data_root)
    
    # Split dataset
    train_data, temp_data = train_test_split(dataset, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    print(dataset.num_classes)
    print(dataset.num_node_features)
    # Initialize model
    model = GNNModel(
        input_dim=dataset.num_node_features,
        hidden_dim=128,
        output_dim=dataset.num_classes,
        num_layers=5,
        dropout=0.2
    ).to(device)
    
    # Train model
    best_epoch, best_f1,best_thres = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=0.001,
        device=device,
        save_dir=save_dir
    )
    
    print(f'\nTraining completed!')
    print(f'Best validation F1: {best_f1:.4f} at epoch {best_epoch}')
    print("Best Threshold : ",best_thres)
    # Load best model and evaluate on test set
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    model.eval()
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            test_predictions.append((out > 0).float().cpu())
            test_labels.append(batch.y.cpu())
    
    test_predictions = torch.cat(test_predictions)
    test_labels = torch.cat(test_labels)
    test_metrics = calculate_metrics(test_predictions, test_labels)
    
    print('\nTest set metrics:')
    for metric, value in test_metrics.items():
        print(f'{metric}: {value:.4f}')

if __name__ == "__main__":
    main()