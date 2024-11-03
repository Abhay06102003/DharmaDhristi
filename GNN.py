import torch
torch.cuda.empty_cache()
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM,AutoModelForSeq2SeqLM,T5ForConditionalGeneration,T5Tokenizer
import torch.nn.functional as F
import networkx as nx
from typing import List, Dict, Tuple, Set
import json
from transformers import pipeline as line
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from gnn_train import GNNModel,WebQSPDataset
from KG import SimplifiedKnowledgeGraph
from dotenv import load_dotenv
load_dotenv()
import os

def visualize_graph(graph):
    # Set the figure size
    plt.figure(figsize=(10, 7))

    # Draw the graph
    pos = nx.spring_layout(graph)  # positions for all nodes
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray')

    # Show the plot
    plt.title("NetworkX Graph Visualization")
    plt.show()

class GNNRagRetriever:
    def __init__(self, 
                 model_path: str,
                 graph_path: str,
                 sentence_encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
                 answer_threshold: float = 0.15,
                 max_paths: int = 15):
        """
        Initialize retriever with pre-trained GNN model
        """
        # Load pre-trained GNN model
        # self.model = GNNModel(
        #     input_dim=768 + 2,  # sentence embedding dim + 2 degree features
        #     hidden_dim=128,
        #     output_dim=1,  # binary classification
        #     num_layers=3,
        #     dropout=0.5
        # ).to('cpu')
        dataset = WebQSPDataset("")
        self.model = GNNModel(
            input_dim=dataset.num_node_features,
            hidden_dim=128,
            output_dim=dataset.num_classes,
            num_layers=5,
            dropout=0.2
        ).to('cpu')
        # Load model weights
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load knowledge graph
        self.graph = nx.read_graphml(graph_path)
        # visualize_graph(self.graph)
        # Initialize sentence encoder
        self.tokenizer = AutoTokenizer.from_pretrained(sentence_encoder_name)
        self.sentence_encoder = AutoModel.from_pretrained(sentence_encoder_name)
        self.sentence_encoder.eval()
        
        # Setup node mappings
        self.node_map = {node: idx for idx, node in enumerate(self.graph.nodes())}
        self.reverse_node_map = {idx: node for node, idx in self.node_map.items()}
        
        self.answer_threshold = answer_threshold
        self.max_paths = max_paths
        
        # Cache tensors
        self.edge_index = None
        self.base_node_features = None
        
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text using sentence transformer"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.sentence_encoder(**inputs)
            # Use mean pooling
            attention_mask = inputs['attention_mask']
            mean_pooling = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), 1) / torch.sum(attention_mask, 1, keepdim=True)
            return mean_pooling
    
    def _networkx_to_geometric(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert NetworkX graph to PyTorch Geometric format"""
        if self.edge_index is not None and self.base_node_features is not None:
            return self.edge_index, self.base_node_features
            
        # Create edge index tensor
        edges = [(self.node_map[src], self.node_map[dst]) 
                for src, dst in self.graph.edges()]
        self.edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Create degree features
        in_degrees = torch.tensor([self.graph.in_degree(node) for node in self.graph.nodes()], dtype=torch.float)
        out_degrees = torch.tensor([self.graph.out_degree(node) for node in self.graph.nodes()], dtype=torch.float)
        
        # Normalize degrees
        in_degrees = in_degrees / (in_degrees.max() + 1e-8)
        out_degrees = out_degrees / (out_degrees.max() + 1e-8)
        
        # Combine degree features
        self.base_node_features = torch.stack([in_degrees, out_degrees], dim=1)
        
        return self.edge_index, self.base_node_features
        
    def _enhance_node_features(self, question_embedding: torch.Tensor) -> torch.Tensor:
        """Enhance node features with question information"""
        _, base_features = self._networkx_to_geometric()
        
        # Encode node labels
        node_texts = [self.reverse_node_map[i] for i in range(len(self.graph))]
        print("-"*100)
        print(node_texts)
        print("-"*100)
        node_embeddings = torch.stack([
            self._encode_text(text) for text in tqdm(node_texts, desc="Encoding nodes")
        ]).squeeze(1)
        
        # Squeeze question embedding
        question_embedding = question_embedding.squeeze(0)

        # Debugging shapes
        print("Node embeddings shape:", node_embeddings.shape)
        print("Question embedding shape:", question_embedding.shape)

        # Concatenate features
        final_node_features = torch.cat((base_features, node_embeddings, question_embedding.unsqueeze(0).expand(node_embeddings.size(0), -1)), dim=1)
        print(base_features.shape)
        print("Final node features shape:", final_node_features.shape)
        
        return final_node_features

    
    def _get_answer_candidates(self, question: str) -> Set[str]:
        """Get answer candidates using GNN predictions"""
        # Encode question
        question_embedding = self._encode_text(question)
        
        # Get enhanced features
        node_features = self._enhance_node_features(question_embedding)
        edge_index, _ = self._networkx_to_geometric()
        data = Data(x=node_features, edge_index=edge_index)
        # Run model inference
        with torch.no_grad():
            probs = self.model(data.x,data.edge_index)
            probs = torch.sigmoid(probs).squeeze()
        print("-"*100)
        print(probs)
        print("-"*100)
        # Get nodes above threshold
        answer_mask = probs > self.answer_threshold
        
        # Convert to original node labels
        answer_candidates = {
            self.reverse_node_map[i] 
            for i in range(len(self.graph)) 
            if answer_mask[i]
        }
        
        return answer_candidates

    def _extract_paths(self, 
                      question_entities: List[str], 
                      answer_candidates: Set[str]) -> List[List[Dict]]:
        """Extract shortest paths between question entities and candidates"""
        all_paths = []
        
        for q_entity in question_entities:
            if q_entity not in self.graph:
                continue
                
            for answer in answer_candidates:
                if answer not in self.graph:
                    continue
                    
                try:
                    path = nx.shortest_path(self.graph, q_entity, answer)
                    
                    path_triples = []
                    for i in range(len(path) - 1):
                        head = path[i]
                        tail = path[i + 1]
                        
                        # Safe edge data access with proper error handling
                        try:
                            # First try to get edge data directly
                            edge_data = self.graph.get_edge_data(head, tail)
                            
                            # If edge_data is None, use default empty dict
                            if edge_data is None:
                                edge_data = {}
                            
                            # If there are multiple edges, take the first one
                            if isinstance(edge_data, dict) and 0 in edge_data:
                                edge_data = edge_data[0]
                            
                            # Get relation with fallbacks
                            relation = (edge_data.get("relation") or 
                                      edge_data.get("type") or 
                                      edge_data.get("label") or 
                                      "related_to")
                            
                        except Exception as e:
                            # Fallback to default relation if any error occurs
                            print(f"Warning: Error accessing edge data between {head} and {tail}: {e}")
                            relation = "related_to"
                        
                        path_triples.append({
                            "head": head,
                            "relation": relation,
                            "tail": tail
                        })
                    
                    all_paths.append(path_triples)
                    
                    if len(all_paths) >= self.max_paths:
                        return all_paths
                        
                except nx.NetworkXNoPath:
                    continue
        
        return all_paths
    
    def _verbalize_paths(self, paths: List[List[Dict]]) -> str:
        """Convert paths to readable format"""
        verbalized = []
        
        for path in paths:
            path_str = ""
            for triple in path:
                if path_str:
                    path_str += " → "
                path_str += f"{triple['head']} → {triple['relation']} → {triple['tail']}"
            verbalized.append(path_str)
            
        return "\n".join(verbalized)


class QAPipeline:
    def __init__(self,
                 gnn_model_path: str,
                 graph_path: str,
                 llm_name: str = "google/flan-t5-large"):
        """Initialize QA pipeline"""
        self.retriever = GNNRagRetriever(gnn_model_path, graph_path)
        self.llm = T5ForConditionalGeneration.from_pretrained("kiri-ai/t5-base-qa-summary-emotion")
        self.tokenizer = T5Tokenizer.from_pretrained("kiri-ai/t5-base-qa-summary-emotion")
        # self.llm =  line("text-generation", model="meta-llama/Llama-3.2-1B",device='cuda',max_new_tokens = 512)     
        
        # Move model to CPU if no GPU available
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.llm = self.llm.to(self.device)
        # self.tokenizer = self.tokenizer.to(self.device)
        # self.llm.eval()
    
    def get_answer(self,question, context):
        input_text = []
        input_text.append(f"q: {question}")
        input_text.append(f"c: {context}")
        input_text = " ".join(input_text)
        features = self.tokenizer([input_text], return_tensors='pt')
        tokens = self.llm.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'], max_length=512)
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        
    def answer_question(self, 
                       question: str,
                       question_entities: List[str]) -> Dict:
        """Process single question"""
        # Get candidates and paths
        answer_candidates = self.retriever._get_answer_candidates(question)
        paths = self.retriever._extract_paths(question_entities, answer_candidates)
        verbalized_paths = self.retriever._verbalize_paths(paths)
        
        # Format prompt
#         prompt = f"""Using the following reasoning paths from a knowledge graph, answer the question.
# Only use information from the provided paths. If you cannot find a definitive answer, say so.

# Reasoning Paths:
# {verbalized_paths}

# Question: {question}

# Answer:"""
        
        # Generate answer
        # with torch.no_grad():
        #     inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        #     outputs = self.llm.generate(
        #         **inputs,
        #         max_length=200,
        #         num_beams=3,
        #         temperature=0.7,
        #         top_p=0.9,
        #         do_sample=True,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #         eos_token_id=self.tokenizer.eos_token_id
        #     )
        #     answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #     answer = answer.replace(prompt, "").strip()
        answer = self.get_answer(question, verbalized_paths)
        return {
            "question": question,
            "reasoning_paths": verbalized_paths,
            "answer_candidates": list(answer_candidates),
            "llm_answer": answer
        }

    def batch_answer_questions(self, 
                             questions: List[str],
                             question_entities: List[List[str]]) -> List[Dict]:
        """Process multiple questions"""
        assert len(questions) == len(question_entities)
        return [self.answer_question(q, e) for q, e in zip(questions, question_entities)]

# Example usage:
if __name__ == "__main__":
    pipeline = QAPipeline(
        gnn_model_path="model_checkpoints/best_model.pt",
        graph_path="graph.gpickle"
    )
    question = "Which fever helen keller have?"
    question_entities = ["helen_keller"]  # From your entity linking system
    
    result = pipeline.answer_question(question, question_entities)
    print("-"*150)
    print(len(result['answer_candidates']))
    print("-"*150)
    print(result['llm_answer'])
    print("-"*150)
    print(result['reasoning_paths'])
    print("-"*150)