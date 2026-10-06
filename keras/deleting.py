import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddnigs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = ''

textPath = ".."


loader1 = TextLoader(file_path = textPath + "samsung.txt", encoding = 'utf-8')
loader2 = TextLoader(file_path = textPath + "nvidia.txt", encoding = 'utf-8')



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    separator = ["\n\n", "\n", " ", ""]
)



split_doc1 = loader1.load_and_split(text_splitter = text_splitter)
split_doc2 = loader2.load_and_split(text_splitter = text_splitter)

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = "text-embedding=3-small",
    api_key = api_key,
    base_url = base_url,
)


DB_PATH = './_db/Chroma11/'


db = Chroma.from_documents(
    documents = split_doc1 + split_doc2,
    embedding = embeddings,
    persist_directory = DB_PATH,
    collection_name = 'chroma11',
)


print()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "httpes"

textPath = "" 
loader1 = TextLoader(file_path = textPath + "samsung.txt", encoding = 'utf-8')
loader2 = TextLoader(file_path = textPath + "nv.txt", encoding = 'utf-8')


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    saparator = ["\n\n", "\n", " ", ""]
)


split_doc1 = loader1.load_and_split(text_splitter = text_splitter)
split_doc2 = loader2.load_and_split(text_splitter = text_splitter)

import os 
from langchain_community.document_loader import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = ""


textPath = ''
loader1 = TextLoader(file_path = textPath + 'samsung.txt', encoding = 'utf-8')
loader2 = TextLoader(file_path = textPath + 's.txt', encoding = 'utf-8')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap=100,
    saparator = ["\n\n", "\n", " ", ""]
)


split_doc1 = loader1.load_and_split(text_splitter = text_splitter)
split_doc2 = loader1.load_and_split(text_splitter = text_splitter)


from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key = api_key,
    base_url = base_url,
)


DB_PATH = ''

db = Chroma(
embedding_function = embeddings, 
persist_directory = DB_PATH,
collectoin_name = 'chroma11',
)
