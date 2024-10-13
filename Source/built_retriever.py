# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from sentence_transformers import SentenceTransformer

path = r"C:\MINE\Hopee\Chatbots_Update\Data\InMemory\vectorstore.csv"

embedding_model = SentenceTransformer("hiieu/halong_embedding")

#embedding data
class CustomEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, query):
        return self.model.encode(query).tolist()

embedding_function = CustomEmbeddingFunction(embedding_model)

loaded_vectorstore = InMemoryVectorStore.load(path, embedding_function)

retriever = loaded_vectorstore.as_retriever()

# #test
# result = retriever.invoke("chương trình tài năng nhằm mục đích gì?")
# print(len(result))
# print(result)