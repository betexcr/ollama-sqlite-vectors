import os
import sqlite3
import sqlite_vec
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(base_url=os.getenv("OLLAMA_URL"), api_key="ollama")
# create in-memory DB and allow it to load extensions
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)

# add vector functionality
sqlite_vec.load(db)


# creates a virtual table, dimensionality of embedding will be 768(dimensionality of 'nomic-embed-text'), auxiliary is for information that is not related to metadata and is handled with the + sign, metadata doesnt need +
db.execute(
    "CREATE VIRTUAL TABLE data USING vec0(embedding float[768], +text TEXT, category TEXT) "
)


def embed_text(text):
    response = client.embeddings.create(
        model=os.getenv("TEXT_EMBED_MODEL",""),  # or "text-embedding-3-small" if using OpenAI
        input=text,
    )
    # Extract the actual float array from the OpenAI/Ollama response object
    return response.data[0].embedding


texts = [
    ("Python is awesome!", "Programming"),
    ("SQLite is awesome!", "Programming"),
    ("Oranges are awesome!", "Fruit"),
    ("Apples are great!", "Fruit"),
    ("I like Windows", "Fruit"),

]


# we need to serialize to the correct float datatype
for text in texts:
    db.execute(
        "INSERT INTO data(text,embedding,category) VALUES (?, ?, ?)",
        (text[0], sqlite_vec.serialize_float32(embed_text(text)), text[1]),
    )

query = "I like bananas!"

# 1. Convert the text string into a vector array of floats
query_embedding = embed_text(query)
# 2. Pass the float array into serialize_float32
for row in db.execute(
    "SELECT text, distance FROM data WHERE embedding MATCH ? AND k = 5 AND category = 'Programming'",
    (sqlite_vec.serialize_float32(query_embedding),),
):
    print(row)
