def add_metadata_tags(docs, content_types):
    for i, doc in enumerate(docs):
        doc.metadata["type"] = content_types[i]
    return docs