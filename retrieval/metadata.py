def add_metadata_tags(docs, content_types):
    print(content_types)
    for i, doc in enumerate(docs):
        print(content_types[i])
        doc.metadata.update(content_types[i])
    return docs