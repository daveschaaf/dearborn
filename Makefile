qdrant-up:
	docker run -p 6333:6333 -p 6334:6333 -v $(shell pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
