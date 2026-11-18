"""Search identical precomputed E5 vectors in three local vector databases."""

import json
from time import monotonic, sleep
from uuid import NAMESPACE_URL, uuid5

import httpx

ENGINES = ("chroma", "qdrant", "weaviate")
BATCH = 64


class ChromaVectors:
    def __init__(self, directory, namespace):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=str(directory / "chroma"), settings=Settings(anonymized_telemetry=False),
        )
        self.name = namespace
        self.collection = None

    def setup(self, _dimensions):
        self.collection = self.client.create_collection(
            self.name, embedding_function=None,
            configuration={"hnsw": {"space": "cosine", "ef_search": 100, "num_threads": 1}},
        )

    def add(self, chunks, vectors):
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks], embeddings=vectors,
            metadatas=[{"source_id": chunk.source_id} for chunk in chunks],
        )

    def count(self):
        return self.collection.count()

    def search(self, vector, limit, source=None):
        result = self.collection.query(query_embeddings=[vector], n_results=limit,
                                       where={"source_id": source} if source else None,
                                       include=[])
        return result["ids"][0]

    def version(self):
        return self.client.get_version()

    def cleanup(self):
        if self.collection is not None:
            self.client.delete_collection(self.name)

    def close(self):
        self.client.close()


class RestVectors:
    def __init__(self, url, name):
        self.client = httpx.Client(base_url=url, timeout=20, trust_env=False)
        self.name = name
        self.created = False

    def req(self, method, path, body=None):
        result = self.client.request(method, path, json=body)
        result.raise_for_status()
        return result.json() if result.content else None

    def wait(self, health):
        deadline = monotonic() + 20
        while True:
            try:
                self.client.get(health).raise_for_status()
                return
            except httpx.HTTPError:
                if monotonic() >= deadline:
                    raise
                sleep(0.25)

    def close(self):
        self.client.close()


def vector_id(chunk):
    return str(uuid5(NAMESPACE_URL, "careops-storage/" + chunk.chunk_id))


class QdrantVectors(RestVectors):
    def __init__(self, namespace):
        super().__init__("http://127.0.0.1:56333", namespace)
        self.path = f"/collections/{namespace}"

    def setup(self, dimensions):
        self.wait("/readyz")
        self.req("PUT", self.path, {"vectors": {"size": dimensions, "distance": "Cosine"}})
        self.created = True
        self.req("PUT", self.path + "/index?wait=true", {
            "field_name": "source_id", "field_schema": "keyword",
        })

    def add(self, chunks, vectors):
        self.req("PUT", self.path + "/points?wait=true", {"points": [
            {"id": vector_id(chunk), "vector": vector,
             "payload": {"chunk_id": chunk.chunk_id, "source_id": chunk.source_id}}
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]})

    def count(self):
        return self.req("POST", self.path + "/points/count", {"exact": True})["result"]["count"]

    def search(self, vector, limit, source=None):
        body = {"query": vector, "limit": limit, "with_payload": True,
                "params": {"exact": True}}
        if source:
            body["filter"] = {"must": [{"key": "source_id", "match": {"value": source}}]}
        result = self.req("POST", self.path + "/points/query", body)
        return [point["payload"]["chunk_id"] for point in result["result"]["points"]]

    def version(self):
        return self.req("GET", "/")["version"]

    def cleanup(self):
        if self.created:
            self.req("DELETE", self.path)


class WeaviateVectors(RestVectors):
    def __init__(self, namespace):
        super().__init__("http://127.0.0.1:58080", "CareOps" + namespace.replace("_", ""))

    def setup(self, _dimensions):
        self.wait("/v1/.well-known/ready")
        self.req("POST", "/v1/schema", {
            "class": self.name, "vectorizer": "none", "vectorIndexType": "flat",
            "vectorIndexConfig": {"distance": "cosine"},
            "properties": [
                {"name": name, "dataType": ["text"], "tokenization": "field"}
                for name in ("chunkKey", "sourceId")
            ],
        })
        self.created = True

    def add(self, chunks, vectors):
        result = self.req("POST", "/v1/batch/objects", {"objects": [
            {"class": self.name, "id": vector_id(chunk), "vector": vector,
             "properties": {"chunkKey": chunk.chunk_id, "sourceId": chunk.source_id}}
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]})
        if len(result) != len(chunks) or any(row["result"]["status"] != "SUCCESS" for row in result):
            raise RuntimeError("Weaviate rejected a vector batch")

    def graphql(self, query):
        response = self.req("POST", "/v1/graphql", {"query": query})
        if response.get("errors"):
            raise RuntimeError(f"Weaviate rejected the query: {response['errors'][0]['message']}")
        return response["data"]

    def count(self):
        rows = self.graphql(f"{{Aggregate{{{self.name}{{meta{{count}}}}}}}}")["Aggregate"][self.name]
        return rows[0]["meta"]["count"]

    def search(self, vector, limit, source=None):
        where = (
            ", where:{path:[\"sourceId\"],operator:Equal,valueText:" + json.dumps(source) + "}"
            if source else ""
        )
        query = (f"{{ Get {{ {self.name}(limit:{limit}, nearVector:{{vector:{json.dumps(vector)}}}"
                 f"{where}) {{ chunkKey }} }} }}")
        rows = self.graphql(query)["Get"][self.name]
        return [row["chunkKey"] for row in rows]

    def version(self):
        return self.req("GET", "/v1/meta")["version"]

    def cleanup(self):
        if self.created:
            self.req("DELETE", f"/v1/schema/{self.name}")


def connect(engine, directory, namespace):
    if engine == "chroma":
        return ChromaVectors(directory, namespace)
    if engine == "qdrant":
        return QdrantVectors(namespace)
    if engine == "weaviate":
        return WeaviateVectors(namespace)
    raise ValueError(f"Unknown vector database {engine}")
