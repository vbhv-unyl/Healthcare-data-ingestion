from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from domain.models import Chunk, RetrievedChunk

_VECTOR_FIELD = "content_vector"
_HNSW_PROFILE = "default-hnsw"
_HNSW_ALGORITHM = "default-hnsw-algorithm"


def create_index_if_not_exists(endpoint: str, api_key: str, index_name: str, vector_dimensions: int) -> None:
    index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
    if index_name in [i.name for i in index_client.list_indexes()]:
        return

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="user_id", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        SimpleField(name="section", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page_start", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="page_end", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="low_confidence", type=SearchFieldDataType.Boolean, filterable=True),
        SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name=_VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name=_HNSW_PROFILE,
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=_HNSW_ALGORITHM)],
        profiles=[VectorSearchProfile(name=_HNSW_PROFILE, algorithm_configuration_name=_HNSW_ALGORITHM)],
    )

    index_client.create_index(SearchIndex(name=index_name, fields=fields, vector_search=vector_search))


def _escape_odata_string(value: str) -> str:
    """OData string literals escape a single quote by doubling it -- same idea as SQL parameterization."""
    return value.replace("'", "''")


class AzureAISearchVectorStore:
    """Implements VectorStorePort."""

    def __init__(self, endpoint: str, api_key: str, index_name: str):
        credential = AzureKeyCredential(api_key)
        self._client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    def upsert_chunks(
        self, doc_id: str, user_id: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        if not chunks:
            return
        documents = [
            {
                "id": f"{doc_id}-{i}",
                "doc_id": doc_id,
                "user_id": user_id,
                "content": chunk.text,
                "section": chunk.section,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "low_confidence": chunk.low_confidence,
                "source_type": chunk.source_type,
                _VECTOR_FIELD: vector,
            }
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self._client.merge_or_upload_documents(documents=documents)

    def delete_document(self, doc_id: str) -> None:
        filter_expr = f"doc_id eq '{_escape_odata_string(doc_id)}'"
        matching_ids = [
            {"id": result["id"]}
            for result in self._client.search(search_text="*", filter=filter_expr, select=["id"])
        ]
        if matching_ids:
            self._client.delete_documents(documents=matching_ids)

    def query(self, query_vector: list[float], user_id: str, top_k: int = 5) -> list[RetrievedChunk]:
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields=_VECTOR_FIELD)
        filter_expr = f"user_id eq '{_escape_odata_string(user_id)}'"

        results = self._client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expr,
            top=top_k,
            select=["content", "section", "page_start", "page_end", "low_confidence", "source_type"],
        )

        return [
            RetrievedChunk(
                text=r["content"],
                section=r["section"],
                page_start=r["page_start"],
                page_end=r["page_end"],
                low_confidence=r["low_confidence"],
                source_type=r["source_type"],
                score=r["@search.score"],
            )
            for r in results
        ]