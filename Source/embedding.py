# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from sentence_transformers import SentenceTransformer
import pandas as pd

path = r"C:\MINE\Hopee\Chatbots_Update\Data\InMemory\vectorstore.csv"
 #load embedding model
embedding_model = SentenceTransformer("hiieu/halong_embedding")

#Load and processing data
file_path = (r"C:\MINE\Hopee\Chatbots_Update\Data\convert.csv")
# loader = CSVLoader(file_path=file_path, encoding='utf-8')
# data = loader.load()
df = pd.read_csv(file_path)
questions = df['Question'].tolist()
answers = [{"id": id_value} for id_value in df['Answer']]
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(data)

#embedding data
class CustomEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, query):
        return self.model.encode(query).tolist()

embedding_function = CustomEmbeddingFunction(embedding_model)

vectorstore = InMemoryVectorStore.from_texts(
    texts=questions, 
    metadatas=answers,
    embedding=embedding_function
)
vectorstore.dump(path)

print("Đã tạo vectorstore!")
