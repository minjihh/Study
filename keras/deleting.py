import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "httpe://"

from glob import glob

path = './_data/rag_data/'