from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

import torch


class SemanticChunking:
    def __init__(self):
        """
        Initialize BERT model for semantic chunking
        """
        self.embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.semantic_chunker = SemanticChunker(self.embed_model,breakpoint_threshold_type = 'percentile',min_chunk_size=512,)
    
    def __call__(self,text):
        """
        Perform semantic chunking on the input text
        """
        chunks = self.semantic_chunker.create_documents([text])
        return chunks

# if __name__ == "__main__":
#     text = """
#     Tom Hanks starred in Forrest Gump, delivering an unforgettable performance as a slow-witted but kind-hearted man who unknowingly influences several defining historical events. The movie was masterfully directed by Robert Zemeckis, who brought the story to life with innovative visual effects and storytelling techniques.

#     Tom Hanks also appeared in Cast Away, which was also directed by Robert Zemeckis. In this survival drama, Hanks portrayed Chuck Noland, a FedEx executive who becomes stranded on an uninhabited island after his plane crashes in the South Pacific. His only companion is a volleyball he names Wilson, which becomes a symbol of his struggle to maintain his sanity in isolation.

#     Forrest Gump was released in 1994 and won several Academy Awards, including Best Picture, Best Director for Zemeckis, and Best Actor for Hanks. The film's success was not just commercial but also cultural, with many of its quotes and scenes becoming iconic parts of cinema history. The movie's soundtrack, featuring songs from different decades, helped establish the historical context of each scene and became a bestseller.

#     Both films showcase Zemeckis's talent for combining human drama with technical innovation, and Hanks's ability to create deeply empathetic characters that resonate with audiences worldwide. These collaborations between Hanks and Zemeckis have become landmarks in American cinema.
#     """    
#     semantic_chunker = SemanticChunking()
#     doc = semantic_chunker(text)
    