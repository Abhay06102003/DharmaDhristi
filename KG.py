import networkx as nx
import torch
from transformers import pipeline
from typing import List, Dict
import matplotlib.pyplot as plt
from py2neo import Graph, Node, Relationship, NodeMatcher
from dotenv import load_dotenv
load_dotenv()
import os

class RebelKGExtractor:
    def __init__(self, model_name: str = "Babelscape/rebel-large"):
        """
        Initialize REBEL model for relation extraction
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.triplet_extractor = pipeline('text2text-generation', model=model_name, tokenizer=model_name, device='cuda')    
        
    def extract_triplets(self, text):
        """
        Extract triplets from text using REBEL
        """
        extracted_text = self.triplet_extractor.tokenizer.batch_decode([self.triplet_extractor(text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])
        text = extracted_text[0]
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
        return triplets

class SimplifiedKnowledgeGraph:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        """Initialize the Simplified Knowledge Graph with REBEL and Neo4j"""
        self.graph = nx.MultiDiGraph()
        self.rebel_extractor = RebelKGExtractor()
        
        # Initialize Neo4j connection
        self.neo4j_graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.node_matcher = NodeMatcher(self.neo4j_graph)
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text by converting to lowercase and replacing spaces with underscores"""
        return text.strip().lower().replace(" ", "_")
        
    def add_triple(self, head: str, relation: str, tail: str):
        """Add a triple to both NetworkX and Neo4j graphs"""
        # Normalize all text
        head = self._normalize_text(head)
        relation = self._normalize_text(relation)
        tail = self._normalize_text(tail)
        
        # Add edge to NetworkX graph
        self.graph.add_edge(head, tail, relation=relation)
        
        # Add to Neo4j
        # Create or get nodes
        head_node = Node("Entity", name=head)
        tail_node = Node("Entity", name=tail)
        
        # Merge nodes (create if doesn't exist)
        self.neo4j_graph.merge(head_node, "Entity", "name")
        self.neo4j_graph.merge(tail_node, "Entity", "name")
        
        # Create relationship
        rel = Relationship(head_node, relation, tail_node,)
        self.neo4j_graph.merge(rel)
        
    def extract_and_add_from_text(self, text: str) -> List[Dict]:
        """Extract relations from text using REBEL and add to both graphs"""
        triplets = self.rebel_extractor.extract_triplets(text)
        
        for triplet in triplets:
            self.add_triple(triplet['head'], triplet['type'], triplet['tail'])
            
        return triplets

    def visualize_graph(self):
        """Visualize the knowledge graph"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=0.9, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=500)
        
        # Draw edges with relation labels
        for (u, v, data) in self.graph.edges(data=True):
            # Draw the edge
            nx.draw_networkx_edges(self.graph, pos, edgelist=[(u, v)], arrows=True)
            
            # Add edge labels
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            plt.text(mid_x, mid_y, data['relation'], fontsize=8, 
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Label nodes
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    class SubgraphRetriever:
        def __init__(self, full_graph: nx.MultiDiGraph, rebel_extractor, llm_model: str = "google/flan-t5-large", hop_size: int = 2):
            """Initialize subgraph retriever with REBEL and LLM"""
            self.full_graph = full_graph
            self.rebel_extractor = rebel_extractor
            self.hop_size = hop_size
            
            # Initialize LLM for question answering
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.llm = pipeline(
                'text2text-generation', 
                model=llm_model, 
                tokenizer=llm_model,
                device=self.device
            )
            
        def _normalize_text(self, text: str) -> str:
            """Normalize text by converting to lowercase and replacing spaces with underscores"""
            return text.strip().lower().replace(" ", "_")
            
        def extract_question_entities(self, question: str) -> List[str]:
            """Extract entities from question using REBEL"""
            triplets = self.rebel_extractor.extract_triplets(question)
            
            # Collect all entities from triplets
            entities = set()
            for triplet in triplets:
                head = self._normalize_text(triplet['head'])
                tail = self._normalize_text(triplet['tail'])
                
                if head in self.full_graph:
                    entities.add(head)
                if tail in self.full_graph:
                    entities.add(tail)
                    
            return list(entities)
        
        def retrieve_subgraph(self, question_entities: List[str]) -> Dict:
            """Retrieve a subgraph around the given entities"""
            # Retrieve subgraph within hop_size
            subgraph_nodes = set()
            for entity in question_entities:
                # Get neighbors within hop_size
                neighbors = list(nx.single_source_shortest_path_length(
                    self.full_graph, 
                    entity, 
                    cutoff=self.hop_size
                ).keys())
                subgraph_nodes.update(neighbors)
            
            # Extract subgraph
            subgraph = self.full_graph.subgraph(subgraph_nodes)
            
            # Prepare subgraph details
            subgraph_details = {
                'nodes': list(subgraph_nodes),
                'edges': []
            }
            
            # Collect edge information
            for u, v, data in subgraph.edges(data=True):
                subgraph_details['edges'].append({
                    'source': u,
                    'target': v,
                    'relation': data['relation']
                })
            
            return subgraph_details
        
        def answer_question(self, question: str) -> str:
            """Answer a natural language question using the knowledge graph and LLM"""
            # Extract entities from the question
            question_entities = self.extract_question_entities(question)
            
            if not question_entities:
                return "Could not find relevant entities in the knowledge graph."
            
            # Retrieve relevant subgraph
            subgraph = self.retrieve_subgraph(question_entities)
            
            # Prepare context for LLM
            context = "Knowledge Graph Context:\n"
            for edge in subgraph['edges']:
                context += f"{edge['source']} {edge['relation']} {edge['target']}\n"
            
            # Combine context with original question for answer generation
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Generate answer using LLM
            answer = self.llm(prompt, max_length=250, num_return_sequences=1)[0]['generated_text']
            
            return answer

def main():
    # Initialize simplified KG with Neo4j connection
    kg = SimplifiedKnowledgeGraph(
        neo4j_uri=os.getenv('NEO4J_URI'),
        neo4j_user=os.getenv('NEO4J_USER'),
        neo4j_password=os.getenv('NEO4J_PASSWORD')  # Replace with your Neo4j password
    )
    
    # Example text for extraction
    text = """
    Tom Hanks starred in Forrest Gump. The movie was directed by Robert Zemeckis.
    Tom Hanks also appeared in Cast Away, which was also directed by Robert Zemeckis.
    Forrest Gump was released in 1994 and won several Academy Awards.
    """
    
    # Extract and add to both graphs
    triplets = kg.extract_and_add_from_text(text)
    print("Extracted triplets:", triplets)
    
    # Initialize subgraph retriever
    retriever = kg.SubgraphRetriever(
        kg.graph,
        kg.rebel_extractor
    )
    
    # Example question
    question = "Who is Director of Forrest Gump?"
    
    # Get answer
    try:
        question_entities = retriever.extract_question_entities(question)
        print(f"Found entities: {question_entities}")
        answer = retriever.answer_question(question)
        print(f"Answer: {answer}")
        kg.visualize_graph()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()