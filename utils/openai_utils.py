from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes.models import *
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery, VectorQuery

from openai import OpenAI
from typing import List, Iterable

class Embedder:
    def __init__(
        self, 
        key=None,
        model="text-embedding-ada-002"
    ):
    
        self.model = model 
        self.client = OpenAI(api_key=key) 
        
    def embed_in_batches(self, texts:List[str], batch_size:str=16):
        
        embeddings = []
        for i in range(0, len(texts), batch_size):            
            batch = texts[i:i+batch_size] 
            response = self.client.embeddings.create(
                input=batch,
                model=self.model 
            )
            embeddings += [item.embedding for item in response.data]
        return embeddings
    
    def embed_single_document(self, text:str):
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
class OpenAIChat:
    
    def __init__(self, 
                 openai_api_key:str,
                 model:str="gpt-3.5-turbo",
                 system_message:str=None,
                 n:int=1,
                 temperature:float=0.2,
                 seed:float=None
                 ):
        self.model = model
        self.system_message = system_message
        self.n = n
        self.temperature = temperature
        self.seed = seed 
        
        # initialize openai client
        self.client = OpenAI(
            api_key=openai_api_key 
        )
        
    def _create_system_message(self, system_message:str=None):
        if system_message==None:    
            system_message = "You are a helpful assistant."
        return {
            "role":"system",
            "content":system_message
        }
        
    def _create_human_message(self, human_message:str):
        return {
            "role":"user",
            "content":human_message
        }    
        
    def _create_assistant_message(self, assistant_message:str):
        return {
            "role":"assistant",
            "content":assistant_message
        }
        
    def __call__(self, prompt:str, memory:List[dict]=None):
        
        if memory == None:
            self.messages = []
            self.messages.append( self._create_system_message(self.system_message) )
        else:
            self.messages = memory
        
        human_message = self._create_human_message(prompt)
        self.messages.append( human_message )
        chat_completion = self.client.chat.completions.create(
            messages = self.messages,
            model=self.model,
            n=self.n,
            temperature=self.temperature,
            seed = self.seed
        )
        response = chat_completion.choices[0].message.content 
        assistant_message = self._create_assistant_message(response)
        self.messages.append(assistant_message)
        memory = self.messages

        return response, memory
    
class RAGBot:
    def __init__(self,
                 fields_to_return:List[str],
                 azure_search_object:'CustomAzureSearch',
                 openai_chat_object:OpenAIChat
                 ):
        
        # list of fields to return from azure ai search (also used in prompt creation)
        self.fields_to_return = fields_to_return
        self.azure_search_object = azure_search_object
        self.openai_chat_object = openai_chat_object
    
    def retrieve_context(self, 
                         query:str) -> Iterable:
        
        search_result = self.azure_search_object.get_results_vector_search(
            query, 
            self.fields_to_return
            )

        return search_result 
    
    def create_prompt(self,
                      query:str,
                      search_result:Iterable) -> str:
        
        # unpack search result and format as string 
        docs = [doc for doc in search_result]
        context_dicts = [{k: d[k] for k in self.fields_to_return if k in d} for d in docs] # keep only requested fields
        self.context_dicts = context_dicts
        context_dicts_strings = [str(doc) for doc in context_dicts]
        context_dicts_strings = "\n\n".join(context_dicts_strings)
        prompt = f"""Answer the query based on the provided context. If the answer is not provided in the context, answer "I don't know".
        
        Context:
        {context_dicts_strings}
        
        Query:
        {query}
        """
        return prompt
    
    def generate_answer(self, prompt:str):
        response, memory = self.openai_chat_object(prompt)
        return response, memory
    
    def __call__(self, query:str):
        search_result = self.retrieve_context(query)
        prompt = self.create_prompt(query, search_result)
        response, memory = self.generate_answer(prompt)
        return response, memory 