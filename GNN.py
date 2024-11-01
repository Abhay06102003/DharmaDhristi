import torch
import torch.nn as nn
from dotenv import load_dotenv
import os
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
from KG import *
load_dotenv()

class GNNRAG:
    class GNNRetriever(nn.Module):
        def __init__(self, hidden_dim: int, num_layers: int = 3):
            """
            GNN-based retriever with configurable depth
            Args:
                hidden_dim: Hidden dimension size
                num_layers: Number of GNN layers (3 for deep GNN, 1 for shallow)
            """
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            # GNN layers
            self.convs = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
            
            # Question encoder
            self.question_encoder = nn.Linear(768, hidden_dim)  # Assuming BERT embeddings
            
            # Answer classifier
            self.classifier = nn.Linear(hidden_dim, 1)  # Binary classification
            
        def forward(self, x, edge_index, question_embedding):
            # Encode question
            q = self.question_encoder(question_embedding)
            
            # Multi-layer GNN processing
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                if i < self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.2, training=self.training)
            
            # Combine node representations with question
            x = x * q.unsqueeze(0)  # Question-aware node representations
            
            # Node classification scores
            scores = self.classifier(x).squeeze(-1)
            return torch.sigmoid(scores)

    def __init__(self, 
                 hidden_dim: int = 256,
                 llm_model: str = "meta-llama/Llama-2-7b-chat-hf",
                 prob_threshold: float = 0.5):
        """
        Initialize GNN-RAG system
        Args:
            hidden_dim: Hidden dimension for GNN
            llm_model: LLM model to use for RAG
            prob_threshold: Probability threshold for answer candidates
        """
        # Initialize deep and shallow GNNs
        self.deep_gnn = self.GNNRetriever(hidden_dim, num_layers=3)
        self.shallow_gnn = self.GNNRetriever(hidden_dim, num_layers=1)
        
        # Initialize LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model,load_in_4bit = True,use_auth_token = os.getenv("HG_token"))
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        
        self.prob_threshold = prob_threshold
        self.entity2idx = {}
        self.idx2entity = {}
        
    def process_question(self, 
                        question: str, 
                        subgraph: nx.MultiDiGraph,
                        question_entities: List[str]) -> Tuple[List[str], List[str]]:
        """
        Process question using GNN-RAG
        Returns:
            answers: List of answer entities
            reasoning_paths: List of verbalized reasoning paths
        """
        # Prepare graph inputs
        x = self._prepare_node_features(subgraph)
        edge_index = self._prepare_edge_index(subgraph)
        q_embed = self._encode_question(question)
        
        # Get predictions from both GNNs
        with torch.no_grad():
            deep_scores = self.deep_gnn(x, edge_index, q_embed)
            shallow_scores = self.shallow_gnn(x, edge_index, q_embed)
        
        # Combine predictions (GNN-RAG+Ensemble)
        combined_scores = torch.maximum(deep_scores, shallow_scores)
        
        # Get candidate answers
        candidate_indices = torch.where(combined_scores > self.prob_threshold)[0]
        candidate_answers = [self.idx2entity[idx.item()] for idx in candidate_indices]
        
        # Extract reasoning paths
        reasoning_paths = self._extract_paths(subgraph, question_entities, candidate_answers)
        verbalized_paths = self._verbalize_paths(reasoning_paths)
        
        # Generate final answer using LLM
        llm_answer = self._generate_llm_answer(question, verbalized_paths)
        
        return llm_answer, verbalized_paths
    
    def _prepare_node_features(self, graph: nx.MultiDiGraph) -> torch.Tensor:
        """Initialize node feature vectors"""
        num_nodes = len(graph.nodes)
        return torch.randn(num_nodes, self.deep_gnn.hidden_dim)
    
    def _prepare_edge_index(self, graph: nx.MultiDiGraph) -> torch.Tensor:
        """Convert graph edges to PyTorch Geometric format"""
        edges = list(graph.edges())
        edge_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t()
        return edge_index
    
    def _encode_question(self, question: str) -> torch.Tensor:
        """Encode question using BERT/other encoder"""
        # Placeholder - replace with actual question encoder
        return torch.randn(768)  # Assuming BERT dimension
    
    def _extract_paths(self, 
                      graph: nx.MultiDiGraph,
                      question_entities: List[str],
                      candidate_answers: List[str]) -> List[List[str]]:
        """Extract shortest paths between question entities and candidate answers"""
        paths = []
        for q_entity in question_entities:
            for answer in candidate_answers:
                try:
                    shortest_paths = nx.all_shortest_paths(graph, q_entity, answer)
                    for path in shortest_paths:
                        full_path = []
                        for i in range(len(path)-1):
                            curr_node = path[i]
                            next_node = path[i+1]
                            relation = graph[curr_node][next_node]['relation']
                            full_path.extend([curr_node, relation])
                        full_path.append(path[-1])
                        paths.append(full_path)
                except nx.NetworkXNoPath:
                    continue
        return paths
    
    def _verbalize_paths(self, paths: List[List[str]]) -> List[str]:
        """Convert paths to natural language format for LLM"""
        verbalized = []
        for path in paths:
            text = ""
            for i in range(0, len(path)-1, 2):
                entity = path[i]
                relation = path[i+1]
                next_entity = path[i+2]
                
                if i == 0:
                    text += f"{entity}"
                text += f" → {relation} → {next_entity}"
            verbalized.append(text)
        return verbalized
    
    def _generate_llm_answer(self, question: str, reasoning_paths: List[str]) -> str:
        """Generate answer using LLM with retrieved paths"""
        prompt = (
            "Based on the reasoning paths, please answer the given question.\n"
            f"Reasoning Paths:\n" + "\n".join(reasoning_paths) + "\n"
            f"Question: {question}"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm.generate(**inputs, max_length=100)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer

    def train(self, 
              train_data: List[Dict],
              num_epochs: int = 10,
              lr: float = 0.001):
        """
        Train GNN retrievers
        Args:
            train_data: List of {question, subgraph, answers} dictionaries
            num_epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer_deep = torch.optim.Adam(self.deep_gnn.parameters(), lr=lr)
        optimizer_shallow = torch.optim.Adam(self.shallow_gnn.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_data:
                # Prepare inputs
                x = self._prepare_node_features(batch['subgraph'])
                edge_index = self._prepare_edge_index(batch['subgraph'])
                q_embed = self._encode_question(batch['question'])
                
                # Prepare target labels
                target = torch.zeros(len(batch['subgraph'].nodes()))
                for ans in batch['answers']:
                    target[self.entity2idx[ans]] = 1
                
                # Train deep GNN
                optimizer_deep.zero_grad()
                deep_scores = self.deep_gnn(x, edge_index, q_embed)
                loss_deep = F.binary_cross_entropy(deep_scores, target)
                loss_deep.backward()
                optimizer_deep.step()
                
                # Train shallow GNN
                optimizer_shallow.zero_grad()
                shallow_scores = self.shallow_gnn(x, edge_index, q_embed)
                loss_shallow = F.binary_cross_entropy(shallow_scores, target)
                loss_shallow.backward()
                optimizer_shallow.step()
                
                total_loss += (loss_deep.item() + loss_shallow.item())
            
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_data):.4f}")
            
if __name__ == "__main__":
    text = """
    Tom Hanks starred in Forrest Gump, delivering an unforgettable performance as a slow-witted but kind-hearted man who unknowingly influences several defining historical events. The movie was masterfully directed by Robert Zemeckis, who brought the story to life with innovative visual effects and storytelling techniques.

    Tom Hanks also appeared in Cast Away, which was also directed by Robert Zemeckis. In this survival drama, Hanks portrayed Chuck Noland, a FedEx executive who becomes stranded on an uninhabited island after his plane crashes in the South Pacific. His only companion is a volleyball he names Wilson, which becomes a symbol of his struggle to maintain his sanity in isolation.

    Forrest Gump was released in 1994 and won several Academy Awards, including Best Picture, Best Director for Zemeckis, and Best Actor for Hanks. The film's success was not just commercial but also cultural, with many of its quotes and scenes becoming iconic parts of cinema history. The movie's soundtrack, featuring songs from different decades, helped establish the historical context of each scene and became a bestseller.

    Both films showcase Zemeckis's talent for combining human drama with technical innovation, and Hanks's ability to create deeply empathetic characters that resonate with audiences worldwide. These collaborations between Hanks and Zemeckis have become landmarks in American cinema.
    """    
    # Using your existing KG
    question = "Who directed Forrest Gump?"
    kg = SimplifiedKnowledgeGraph(
        neo4j_uri=os.getenv('NEO4J_URI'),
        neo4j_user=os.getenv('NEO4J_USER'),
        neo4j_password=os.getenv('NEO4J_PASSWORD') 
    )    
    retriever = kg.SubgraphRetriever(
        kg.graph,
        kg.rebel_extractor
    )
    gnn_rag = GNNRAG(hidden_dim=256)
    question_entities = retriever.extract_question_entities(question)
    subgraph = retriever.retrieve_subgraph(question_entities)
    answer, paths = gnn_rag.process_question(question, subgraph, question_entities)