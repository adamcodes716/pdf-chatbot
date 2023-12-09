import pinecone
from dotenv import load_dotenv
import os
import feedparser
import numpy as np
import openai
import requests
from bs4 import BeautifulSoup

load_dotenv()

# pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENV_NAME'])
# active_indexes = pinecone.list_indexes()

# OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY'] 
# get the Pinecone API key and environment
pinecone_api = os.environ['PINECONE_API_KEY']
pinecone_env = os.environ['PINECONE_ENV_NAME']
 
pinecone.init(api_key=pinecone_api, environment=pinecone_env)


# set index; must exist
index = pinecone.Index('lang-chain-index')
 
# URL of the RSS feed to parse
url = 'https://blog.baeke.info/feed/'
 
# Parse the RSS feed with feedparser
feed = feedparser.parse(url)
 
# get number of entries in feed
entries = len(feed.entries)
print("Number of entries: ", entries)
 
post_texts = []
pinecone_vectors = []
for i, entry in enumerate(feed.entries[:50]):
    # report progress
    print("Processing entry ", i, " of ", entries)
 
    r = requests.get(entry.link)
    soup = BeautifulSoup(r.text, 'html.parser')
    article = soup.find('div', {'class': 'entry-content'}).text
 
    # vectorize with OpenAI text-emebdding-ada-002
    embedding = openai.Embedding.create(
        input=article,
        model="text-embedding-ada-002"
    )
 
    # print the embedding (length = 1536)
    vector = embedding["data"][0]["embedding"]
 
    # append tuple to pinecone_vectors list
    pinecone_vectors.append((str(i), vector, {"url": entry.link}))
 
# all vectors can be upserted to pinecode in one go
upsert_response = index.upsert(vectors=pinecone_vectors)
 
print("Vector upload complete.")


#print (active_indexes)
# index = pinecone.Index("lang-chain-index")

# upsert_response = index.upsert(
#     vectors=[
#         (
#          "vec1",                # Vector ID 
#          [0.1, 0.2, 0.3, 0.4],  # Dense vector values
#          {"genre": "drama"}     # Vector metadata
#         ),
#         (
#          "vec2", 
#          [0.2, 0.3, 0.4, 0.5], 
#          {"genre": "action"}
#         )
#     ]
# )