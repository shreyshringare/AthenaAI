"""RAG (retrieval-augmented generation) pipeline for AthenaAI.

HOW IT FITS IN THE SYSTEM:
This package turns raw text into searchable, embedded chunks and back into
ranked results for a query. The modules form a linear ingest pipeline plus a
separate query-side path:

    ingest:  loader.DocumentLoader
                 -> parser.DocumentParser      (raw text -> ParsedDocument)
                 -> chunker.SlidingWindowChunker (ParsedDocument -> list[Chunk])
                 -> embedder.CloudEmbedder/MockEmbedder (texts -> vectors)
                 -> retriever.PgVectorRetriever  (store chunks + vectors)

    query:   retriever.PgVectorRetriever.search (embed query -> ANN search)
                 -> reranker.CosineReranker      (re-order candidates)

`athenai.gateway.app` wires a `DocumentLoader` into `app.state.document_loader`
when `ATHENA_DB_URL` and `ATHENA_EMBEDDER_URL` are both set. `athenai.gateway.routes`
calls `DocumentLoader.ingest()` for `/v1/documents/ingest` and
`DocumentLoader.search()` (which wraps retriever.search() + reranker.rerank())
for `/v1/documents/search`.
"""
