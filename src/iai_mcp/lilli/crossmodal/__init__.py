"""Embedding <-> HV bridge. from_embedding(emb[384]) -> bytes via SHA256-locked projection.
to_embedding_neighbors(hv) -> list[UUID] via Hippo hnswlib lookup."""
