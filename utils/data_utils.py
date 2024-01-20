
class Chunker():

    def __init__(
        self,
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len,
        is_separator_regex = False
    ):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = length_function,
            is_separator_regex = is_separator_regex
        )
        
    
    def chunk_text(self, text):
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    
class Embedder:
    
    def __init__(
        self, 
        key:str,
        model:str="text-embedding-ada-002"
    ):
        import os
        import math
        from langchain_openai import OpenAIEmbeddings
        self.embedder = OpenAIEmbeddings(openai_api_key=key, model=model)
        
    def embed_in_batches(self, chunks, batch_size=16):
        num_batches = math.ceil(len(chunks) / batch_size)
        embeddings = []
        for i in range(num_batches):            
            batch = chunks[i*batch_size:i*batch_size+batch_size]
            embeddings_batch = self.embedder.embed_documents(batch)
            embeddings += embeddings_batch
        return embeddings
    

class PDFExtractor:
    
    @staticmethod    
    def extract_text_from_pdf(pdf_data):
        """
        Extracts text from PDF bytes.

        :param pdf_data: The binary content of the PDF file.
        :return: Extracted text as a single string.
        """
        import fitz  # PyMuPDF
        text = ""
        with fitz.open(pdf_data, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()

        return text
