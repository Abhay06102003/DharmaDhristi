import networkx as nx
import torch
torch.cuda.empty_cache()
from transformers import pipeline
from typing import List, Dict
import matplotlib.pyplot as plt
from py2neo import Graph, Node, Relationship, NodeMatcher
import PyPDF2
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
import os
from semantic_chunking import SemanticChunking

def extract_text_from_pdf(pdf_path):
    # Open the PDF file in binary mode
    with open(pdf_path, "rb") as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Initialize an empty string to store the extracted text
        full_text = ""
        
        # Iterate through each page and extract text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            full_text += page.extract_text()
        
    return full_text

def save_list_to_file(list_data, filename):
    with open(filename, "wb") as file:
        pickle.dump(list_data, file)

# Load list from file
def load_list_from_file(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

class RebelKGExtractor:
    def __init__(self, model_name: str = "Babelscape/mrebel-large", max_length: int = 1024):
        """
        Initialize REBEL model for relation extraction
        Args:
            model_name: The name of the REBEL model to use
            max_length: Maximum sequence length for processing
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.triplet_extractor = pipeline(
            'translation_xx_to_yy',
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            max_length=self.max_length
        )
    
    def _split_text(self, text: str, max_chars: int = 450) -> List[str]:
        """
        Split text into smaller chunks based on sentence boundaries
        Args:
            text: Input text to split
            max_chars: Maximum characters per chunk
        Returns:
            List of text chunks
        """
        # Split text into sentences (basic implementation)
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_chars and current_chunk:
                # Join the current chunk and add to chunks list
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
        
    def extract_triplets(self, text: str) -> List[Dict]:
        """
        Extract triplets from text, handling long sequences by splitting into chunks
        Args:
            text: Input text to extract triplets from
        Returns:
            List of triplets extracted from the text
        """
        # Split text into smaller chunks
        if len(text) > self.max_length:
            text_chunks = self._split_text(text)
        else:
            text_chunks = [text]
        text_chunks = self._split_text(text)
        all_triplets = []
        
        # Process each chunk
        for chunk in text_chunks:
            try:
                # Extract triplets from the current chunk
                chunk_result = self.triplet_extractor.tokenizer.batch_decode([
                    self.triplet_extractor(
                        chunk,
                        decoder_start_token_id=250058,
                        src_lang="en_XX",
                        tgt_lang="<triplet>",
                        return_tensors=True,
                        return_text=False,
                        max_length=self.max_length
                    )[0]["translation_token_ids"]
                ])
                
                text = chunk_result[0]
                triplets = []
                relation = ''
                text = text.strip()
                current = 'x'
                subject, relation, object_, object_type, subject_type = '','','','',''

                for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__en__", "").split():
                    if token == "<triplet>" or token == "<relation>":
                        current = 't'
                        if relation != '':
                            triplets.append({
                                'head': subject.strip(),
                                'head_type': subject_type,
                                'type': relation.strip(),
                                'tail': object_.strip(),
                                'tail_type': object_type
                            })
                            relation = ''
                        subject = ''
                    elif token.startswith("<") and token.endswith(">"):
                        if current == 't' or current == 'o':
                            current = 's'
                            if relation != '':
                                triplets.append({
                                    'head': subject.strip(),
                                    'head_type': subject_type,
                                    'type': relation.strip(),
                                    'tail': object_.strip(),
                                    'tail_type': object_type
                                })
                            object_ = ''
                            subject_type = token[1:-1]
                        else:
                            current = 'o'
                            object_type = token[1:-1]
                            relation = ''
                    else:
                        if current == 't':
                            subject += ' ' + token
                        elif current == 's':
                            object_ += ' ' + token
                        elif current == 'o':
                            relation += ' ' + token
                
                if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
                    triplets.append({
                        'head': subject.strip(),
                        'head_type': subject_type,
                        'type': relation.strip(),
                        'tail': object_.strip(),
                        'tail_type': object_type
                    })
                
                all_triplets.extend(triplets)
                
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
        
        return all_triplets

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
        rel = Relationship(head_node, relation, tail_node)
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
    # text = """
    # Tom Hanks starred in Forrest Gump, delivering an iconic performance that earned him an Academy Award for Best Actor. The movie was masterfully directed by Robert Zemeckis, who brought the heartwarming story to life. The film follows the extraordinary life journey of Forrest Gump, a slow-witted but kind-hearted man from Alabama.

    # Tom Hanks also appeared in Cast Away, which was also directed by Robert Zemeckis. In Cast Away, Hanks portrayed Chuck Noland, a FedEx executive who survives a plane crash and becomes stranded on an uninhabited island for four years. This challenging role showcased Hanks' incredible range as an actor.

    # Forrest Gump was released in 1994 and won several Academy Awards, including Best Picture, Best Director, and Best Adapted Screenplay. The film's groundbreaking visual effects seamlessly integrated Forrest into historical footage with figures like John F. Kennedy and John Lennon. The movie's soundtrack became a cultural phenomenon, featuring classic songs from multiple decades.

    # The film's success extended beyond awards, as it became a box office sensation, grossing over $678 million worldwide. Robin Wright played Jenny Curran, Forrest's childhood friend and love interest, while Gary Sinise portrayed Lieutenant Dan Taylor, Forrest's platoon leader in Vietnam who later becomes his business partner in the shrimping industry.

    # The movie's memorable quotes, such as "Life is like a box of chocolates" and "Run, Forrest, run!" became deeply embedded in popular culture. The story spans several decades of American history, touching on pivotal moments like the Vietnam War, the Watergate scandal, and the emergence of Apple Computer.

    # Sally Field played Mrs. Gump, Forrest's devoted mother who goes to great lengths to ensure her son receives a proper education. The film also features Michael Conner Humphreys as young Forrest, whose real-life accent inspired Tom Hanks' portrayal of the adult character.

    # The screenplay was adapted by Eric Roth from the 1986 novel of the same name by Winston Groom. The film's production took place primarily in South Carolina, Georgia, and North Carolina, with the famous running scenes filmed across multiple locations in America. The iconic bench scenes were filmed in Chippewa Square in Savannah, Georgia.

    # Alan Silvestri composed the film's emotional musical score, which perfectly complemented the story's touching moments. The movie's success influenced popular culture and spawned a restaurant chain, Bubba Gump Shrimp Company, named after the film's fictional shrimping business.
    # """
    if os.path.exists("Helen_chunk.pkl"):
        print("Loading pre-chunked data from Helen_chunk.pkl ...")
        docs = load_list_from_file("Helen_chunk.pkl")
    else:
        text = extract_text_from_pdf("Helen Keller.pdf")
        print("PDF Loaded ...")
        Semantic_chunking = SemanticChunking()
        docs = Semantic_chunking(text=text)
        save_list_to_file(docs, "Helen_chunk.pkl")
        print("Semantic Chunking Done ...")

    # Process documents with a progress bar
    for doc in tqdm(docs, desc="Processing Docs"):
        content = doc.page_content
        # Extract and add to both graphs
        _ = kg.extract_and_add_from_text(content)

    nx.write_graphml(kg.graph, 'graph.gpickle')
    print("KG Created!!")
    # print("Extracted triplets:", triplets)
    
    # Initialize subgraph retriever
    # retriever = kg.SubgraphRetriever(
    #     kg.graph,
    #     kg.rebel_extractor
    # )
    
    # # Example question
    # question = "When forrest gump released?"
    
    # # Get answer
    # try:
    #     question_entities = retriever.extract_question_entities(question)
    #     print(f"Found entities: {question_entities}")
    #     answer = retriever.answer_question(question=question)
    #     print(answer)
    #     kg.visualize_graph()
    # except Exception as e:
    #     print(f"Error: {e}")

if __name__ == "__main__":
    main()