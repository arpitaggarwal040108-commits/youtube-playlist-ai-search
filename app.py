from flask import Flask, render_template, request
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google import genai
from google.genai import types

import os
import pickle
import faiss
import numpy as np

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

youtube = build(
    "youtube",
    "v3",
    developerKey=YOUTUBE_API_KEY
)

client = genai.Client(
    api_key=GEMINI_API_KEY
)
def get_playlist_id(url):

    if "list=" not in url:
        return None
    playlist_id = url.split("list=")[1]
    if "&" in playlist_id:
        playlist_id = playlist_id.split("&")[0]

    return playlist_id

def get_embedding(text):

    if not text.strip():
        return None

    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
        )
        return response.embeddings[0].values

    except Exception as e:
        print("Embedding Error:", e)
        return None

def get_playlist_videos(playlist_id):
    videos = []
    next_page_token = None
    while True:

        response = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token

        ).execute()
        items = response.get("items", [])

        for item in items:
            snippet = item.get("snippet", {})
            if "resourceId" not in snippet:
                continue

            video_id = snippet["resourceId"]["videoId"]
            thumbnails = snippet.get("thumbnails", {})
            thumbnail = (
                thumbnails.get("high", {}).get("url")
                or thumbnails.get("medium", {}).get("url")
                or thumbnails.get("default", {}).get("url")
                or ""
            )

            title = snippet.get("title", "")
            description = snippet.get("description", "")
            channel = snippet.get(
                "videoOwnerChannelTitle",
                "Unknown"
            )
            published = snippet.get(
                "publishedAt",
                ""
            )

            embedding_text = f"""
Title:
{title}

Description:
{description}

Channel:
{channel}
"""

            embedding = get_embedding(
                embedding_text
            )

            print(
                f"Processing: {title}"
            )

            print(
                "Embedding:",
                embedding is not None
            )

            videos.append({

                "video_id": video_id,
                "title": title,
                "description": description,
                "channel": channel,
                "published_at": published,
                "thumbnail": thumbnail,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "embedding": embedding

            })
        next_page_token = response.get(
            "nextPageToken"
        )
        if not next_page_token:
            break
    return videos

def build_faiss_index(videos):

    embeddings = []
    metadata = []

    for video in videos:

        if video["embedding"] is not None:

            embeddings.append(video["embedding"])
            metadata.append(video)

    if len(embeddings) == 0:

        print("No embeddings generated.")
        return False

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    os.makedirs(
        "vector_db",
        exist_ok=True
    )

    faiss.write_index(
        index,
        "vector_db/youtube.index"
    )

    with open(
        "vector_db/metadata.pkl",
        "wb"
    ) as f:

        pickle.dump(metadata, f)
    print("FAISS Index Created Successfully")
    return True

def search_videos(query):

    if not os.path.exists(
        "vector_db/youtube.index"
    ):
        return []

    if not os.path.exists(
        "vector_db/metadata.pkl"
    ):
        return []
    embedding = get_embedding(query)
    if embedding is None:
        return []

    embedding = np.array(
        [embedding],
        dtype=np.float32
    )

    try:
        index = faiss.read_index(
            "vector_db/youtube.index"
        )

        with open(
            "vector_db/metadata.pkl",
            "rb"
        ) as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(e)
        return []
    distances, indices = index.search(
        embedding,
        5
    )

    results = []
    for i in indices[0]:
        if i < len(metadata):
            results.append(metadata[i])

    return results

@app.route("/", methods=["GET", "POST"])
def home():

    message = ""

    playlist_url = ""

    videos = []

    search_results = []

    if request.method == "POST":

        playlist_url = request.form.get(
            "playlist_url",
            ""
        ).strip()

        search_query = request.form.get(
            "search_query",
            ""
        ).strip()

        if playlist_url:

            playlist_id = get_playlist_id(
                playlist_url
            )
            if playlist_id:

                try:
                    videos = get_playlist_videos(
                        playlist_id
                    )
                    success = build_faiss_index(
                        videos
                    )

                    if success:
                        message = (
                            f"{len(videos)} videos indexed successfully."
                        )

                    else:
                        message = (
                            "Failed to generate embeddings."
                        )

                except Exception as e:
                    print(e)
                    message = (
                        "Error while indexing playlist."
                    )

            else:

                message = (
                    "Invalid Playlist URL."
                )

        elif search_query:
            search_results = search_videos(
                search_query
            )

            if len(search_results):
                message = (
                    f"{len(search_results)} Results Found"
                )

            else:
                message = (
                    "No Results Found"
                )

    return render_template(
        "index.html",
        playlist_url=playlist_url,
        videos=videos,
        search_results=search_results,
        message=message

    )

if __name__ == "__main__":

    app.run(
        debug=True
    )