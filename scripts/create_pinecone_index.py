"""One-time script to create the Pinecone index for the Palm RAG backend."""

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="pcsk_Bmn8U_LCyRcCUAJvWYb5E5TzTajCp4FfNsxKXEzGMCUhVNnujJRXmY4oeSQMdotDRAZ5L")

INDEX_NAME = "palm-rag"

# Check if index already exists
existing_indexes = [idx.name for idx in pc.list_indexes()]
print(f"Existing indexes: {existing_indexes}")

if INDEX_NAME in existing_indexes:
    print(f"✅ Index '{INDEX_NAME}' already exists!")
    desc = pc.describe_index(INDEX_NAME)
    print(f"   Dimension: {desc.dimension}")
    print(f"   Metric: {desc.metric}")
    print(f"   Host: {desc.host}")
else:
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # text-embedding-3-small output dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )
    print(f"✅ Index '{INDEX_NAME}' created successfully!")
    desc = pc.describe_index(INDEX_NAME)
    print(f"   Host: {desc.host}")
    print(f"\n⚠️  Update PINECONE_HOST in your .env if needed:")
    print(f"   PINECONE_HOST={desc.host}")
