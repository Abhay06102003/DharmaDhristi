import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx

class DeepGNNProcessor:
    class DeepGNN(nn.Module):
        def __init__(self, hidden_dim, num_layers):
            super().__init__()
            self.num_layers = num_layers
            self.convs = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
            self.classifier = nn.Linear(hidden_dim, 2)  # Binary classification (answer vs non-answer)
            
        def forward(self, x, edge_index):
            # Multi-layer GNN processing
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                if i < self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.2, training=self.training)
            
            # Final node classifications
            scores = self.classifier(x)
            return F.softmax(scores, dim=1)
    def __init__(self, hidden_dim=256, num_layers=3):
        """
        Initialize Deep GNN processor for KGQA
        Args:
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers (using 3 for deep GNN)
        """
        self.model = self.DeepGNN(hidden_dim, num_layers)
        self.entity2idx = {}  # Entity to index mapping
        self.idx2entity = {}  # Index to entity mapping
        self.relation2idx = {}  # Relation to index mapping
        

    def process_subgraph(self, subgraph, question_entities):
        """
        Process subgraph using Deep GNN and extract reasoning paths
        Args:
            subgraph: NetworkX graph representing KG subgraph
            question_entities: List of entity IDs from the question
        Returns:
            candidate_answers: List of candidate answer entities
            reasoning_paths: List of paths connecting question entities to answers
        """
        # Convert graph to PyTorch Geometric format
        x = self._prepare_node_features(subgraph)
        edge_index = self._prepare_edge_index(subgraph)
        
        # Run GNN inference
        with torch.no_grad():
            node_scores = self.model(x, edge_index)
        
        # Get candidate answers (nodes with high probability scores)
        answer_probs = node_scores[:, 1]  # Probability of being an answer
        candidate_indices = torch.where(answer_probs > 0.5)[0]
        candidate_answers = [self.idx2entity[idx.item()] for idx in candidate_indices]
        
        # Extract shortest paths
        reasoning_paths = []
        for q_entity in question_entities:
            for answer in candidate_answers:
                paths = self._extract_shortest_paths(subgraph, q_entity, answer)
                reasoning_paths.extend(paths)
                
        return candidate_answers, reasoning_paths
    
    def verbalize_paths(self, reasoning_paths):
        """
        Convert reasoning paths to natural language format
        Args:
            reasoning_paths: List of paths (each path is list of [entity, relation, entity, ...])
        Returns:
            List of verbalized paths
        """
        verbalized_paths = []
        for path in reasoning_paths:
            verbalized_path = ""
            for i in range(0, len(path)-1, 2):
                entity = path[i]
                relation = path[i+1]
                next_entity = path[i+2]
                
                if i == 0:
                    verbalized_path += f"{entity}"
                verbalized_path += f" → {relation} → {next_entity}"
            
            verbalized_paths.append(verbalized_path)
            
        return verbalized_paths
    
    def _prepare_node_features(self, graph):
        """Initialize node feature vectors"""
        num_nodes = len(graph.nodes)
        return torch.randn(num_nodes, self.model.hidden_dim)
    
    def _prepare_edge_index(self, graph):
        """Convert graph edges to PyTorch Geometric format"""
        edges = list(graph.edges())
        edge_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t()
        return edge_index
    
    def _extract_shortest_paths(self, graph, source, target):
        """Extract all shortest paths between source and target nodes"""
        try:
            paths = list(nx.all_shortest_paths(graph, source, target))
            # Convert paths to include relations
            full_paths = []
            for path in paths:
                full_path = []
                for i in range(len(path)-1):
                    curr_node = path[i]
                    next_node = path[i+1]
                    relation = graph[curr_node][next_node]['relation']
                    full_path.extend([curr_node, relation])
                full_path.append(path[-1])
                full_paths.append(full_path)
            return full_paths
        except nx.NetworkXNoPath:
            return []