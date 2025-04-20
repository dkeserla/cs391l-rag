def add_metadata_tags(docs, content_type="text"):
    for doc in docs:
        doc.metadata["type"] = content_type
    return docs