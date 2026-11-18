"""Use the same E5 vectors in the LangChain and LlamaIndex vector stores."""

from careops.retrieval import Hit


class LangChainIndex:
    def __init__(self, dense):
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
        from langchain_core.vectorstores import InMemoryVectorStore

        vectors = {chunk.text: vector.tolist() for chunk, vector in zip(
            dense.chunks, dense.matrix, strict=True
        )}

        class LocalEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [vectors[text] for text in texts]

            def embed_query(self, text):
                return dense.encoder.query(text).tolist()

        self.store = InMemoryVectorStore(LocalEmbeddings())
        self.chunks = {chunk.chunk_id: chunk for chunk in dense.chunks}
        documents = [
            Document(page_content=chunk.text, metadata={"chunk_id": chunk.chunk_id})
            for chunk in dense.chunks
        ]
        self.store.add_documents(documents, ids=list(self.chunks))
        self.payload_bytes = dense.matrix.size * 8

    def search(self, query, top_k):
        return [
            Hit(self.chunks[doc.metadata["chunk_id"]], float(score))
            for doc, score in self.store.similarity_search_with_score(query, k=top_k)
        ]


class LlamaIndex:
    def __init__(self, dense):
        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores import SimpleVectorStore

        self.encoder = dense.encoder
        self.chunks = {chunk.chunk_id: chunk for chunk in dense.chunks}
        self.store = SimpleVectorStore()
        self.store.add([
            TextNode(id_=chunk.chunk_id, text=chunk.text, embedding=vector.tolist())
            for chunk, vector in zip(dense.chunks, dense.matrix, strict=True)
        ])
        self.payload_bytes = dense.matrix.size * 8

    def search(self, query, top_k):
        from llama_index.core.vector_stores.types import VectorStoreQuery

        result = self.store.query(VectorStoreQuery(
            query_embedding=self.encoder.query(query).tolist(),
            similarity_top_k=top_k,
        ))
        return [
            Hit(self.chunks[key], float(score))
            for key, score in zip(result.ids, result.similarities, strict=True)
        ]
