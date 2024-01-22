from typing import List, Iterable 
from .openai_utils import Embedder
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.models import VectorizedQuery, VectorQuery

class CustomAzureSearch:
    def __init__(self,
                 searchservice:str,
                 searchkey:str,
                 index_name:str,
                 number_results_to_return:int,
                 number_near_neighbors:int,
                 embedding_field_name:str,
                 openai_api_key:str,
                 embedding_model:str
                 ):
        
        self.number_results_to_return=number_results_to_return
        self.number_near_neighbors=number_near_neighbors
        self.embedding_field_name=embedding_field_name
        self.openai_api_key=openai_api_key
        self.embedding_model=embedding_model
        
        # initialize search client
        self.search_client = SearchClient(
            endpoint="https://{}.search.windows.net/".format(searchservice),
            index_name=index_name,
            credential=AzureKeyCredential(searchkey)
        )
        
    def get_results_vector_search(self, 
                                  query:str, 
                                  fields_to_return:List[str]=None):
        
        vector_query = self.get_vectorized_query(query)
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=fields_to_return,
            top=self.number_results_to_return
        )
        return results
        
    def get_vectorized_query(self, query:str):
        query_vector = self.get_embedding_query_vector(query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=self.number_near_neighbors,
            fields=self.embedding_field_name
        )
        return vector_query 
    
    def get_embedding_query_vector(self, query:str):
        embedder = Embedder(key=self.openai_api_key, model=self.embedding_model)
        query_vector = embedder.embed_single_document(query)
        return query_vector 
    
    
def create_search_index(index_name:str, 
                        searchservice:str, 
                        key:str,
                        fields:List[object],
                        vector_search:object,
                        semantic_search:object
                        ):
    
    # Initialize search index client
    print(f"Ensuring search index {index_name} exists")
    index_client = SearchIndexClient(
        endpoint=f"https://{searchservice}.search.windows.net/",
        credential=AzureKeyCredential(key)
    )
    
    # define fields if index does not exist
    if index_name not in index_client.list_index_names():
        
        print(f"Creating index {index_name}")
  
        # create the search index with vector and semantic settings
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search 
        )
        result = index_client.create_or_update_index(index)
        print(f"{result.name} created")
        
    else:
        print(f"Search index {index_name} already exists")
        
class PDFExtractor:
       
    @staticmethod
    def get_document_text(pdf_path):
        """ 
        Extracts text from pdf and returns text and page map
        
        :param pdf_path: Location of pdf file
        :return: page_text: String of concatenated text
        :return: page_map: dictionary mapping offset to page number
        """
        from pypdf import PdfReader
        offset = 0
        all_text = ""
        page_map = {}
        reader = PdfReader(pdf_path)
        pages = reader.pages 
        for page_num, page in enumerate(pages):
            page_text = page.extract_text()
            all_text += page_text
            page_map[offset] = page_num
            offset += len(page_text.split(' '))
        return all_text, page_map
    
class Chunker():

    def __init__(
        self,
        chunk_size = 500,
        overlap = 50,
        separator = " " 
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap 
        self.separator = separator
    
    def split_text(self, text):
        splits = text.split(' ')
        return splits
            
    def get_chunk_indices(self, text):
        splits = self.split_text(text)
        chunk_indices = [ (i, i+self.chunk_size) for i in range(0, len(splits)-self.overlap, self.chunk_size-self.overlap)]
        return chunk_indices       
    
    def create_chunks(self, text, page_map):
        
        def find_page(start_idx, page_map):
            offset = max(x for x in page_map.keys() if x <= start_idx)
            page_num = page_map[offset]
            return page_num
        
        chunks = []
        chunk_indices = self.get_chunk_indices(text)            
        for start_idx, end_idx in chunk_indices:
            page_num = find_page(start_idx, page_map)
                
            chunk = text.split(self.separator)[start_idx:end_idx]
            chunk = self.separator.join(chunk)
            chunks.append( (chunk, page_num))
        
        return chunks
        
def create_sections(chunks, embedder, sourcefile):    
    for i, (text, page_num) in enumerate(chunks):
        yield {
            "id":str(i),
            "content":text,
            "embedding":embedder.embed_single_document(text),
            "sourcepage":str(page_num),
            "sourcefile":sourcefile
        }
        
class Uploader:
    def __init__(self, searchservice:str, index_name:str, key:str):
        
        self.client = SearchClient(
            endpoint="https://{}.search.windows.net/".format(searchservice),
            index_name=index_name, 
            credential=AzureKeyCredential(key)
        )
    
    def upload_documents(self, sections:Iterable):
        """ 
        Can refactor to upload in batches
        """
        from tqdm import tqdm 
        sections_unpacked = [section for section in sections]
        for section in tqdm(sections_unpacked, total=len(sections_unpacked)):
            _ = self.client.upload_documents(documents=[section])
        print(f"Documents uploaded")