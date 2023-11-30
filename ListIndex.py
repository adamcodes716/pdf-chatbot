import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENV_NAME'])
active_indexes = pinecone.list_indexes()
print (active_indexes)
index = pinecone.Index("docs-forever")

upsert_response = index.upsert(
    vectors=[
        (
         "vec1",                # Vector ID 
         [0.1, 0.2, 0.3, 0.4],  # Dense vector values
         {"genre": "drama"}     # Vector metadata
        ),
        (
         "vec2", 
         [0.2, 0.3, 0.4, 0.5], 
         {"genre": "action"}
        )
    ]
)