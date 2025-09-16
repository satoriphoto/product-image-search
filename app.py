# app.py
import streamlit as st
import os
import io
import numpy as np
from PIL import Image
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import tensorflow as tf
import tensorflow_hub as hub

st.set_page_config(page_title="Product Image Search", layout="wide")
st.title("📸 AI Product Image Finder (Google Drive)")

# ---------------------------
# CONFIG: Set your All Photos folder ID here
# Example Drive link: https://drive.google.com/drive/folders/1Ni0pf2VKo_U4Pe6falRWX-FqL6CX_mZj
# Folder ID = the long string after /folders/
ROOT_FOLDER_ID = "1Ni0pf2VKo_U4Pe6falRWX-FqL6CX_mZj"  # <-- REPLACE with your folder id

# ---------------------------
# Authenticate with service account from Streamlit secrets
# Put the service account JSON into Streamlit Secrets under key "gcp_service_account"
@st.cache_resource
def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)

drive_service = get_drive_service()

# ---------------------------
# List files recursively in Drive folder
def list_files_recursive(folder_id):
    results = []
    query = f"'{folder_id}' in parents and trashed=false"
    page_token = None
    while True:
        response = drive_service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()

        for file in response.get("files", []):
            if file["mimeType"] == "application/vnd.google-apps.folder":
                # recurse into subfolder
                results.extend(list_files_recursive(file["id"]))
            elif file["mimeType"].startswith("image/"):
                results.append(file)

        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break
    return results

# ---------------------------
# Feature extraction model (TF Hub)
@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    return hub.load(model_url)

model = load_model()

def preprocess_image(img):
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def get_embedding(img: Image.Image):
    img_arr = preprocess_image(img.convert("RGB"))
    emb = model(img_arr)
    return np.array(emb).reshape(-1)

# ---------------------------
# Build index (first run) — this can take time for thousands of images
@st.cache_resource
def build_index():
    st.info("🔍 Indexing images in Google Drive (this may take a while on first run)...")
    files = list_files_recursive(ROOT_FOLDER_ID)
    embeddings, ids, names = [], [], []
    for f in files:
        try:
            request = drive_service.files().get_media(fileId=f["id"])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.seek(0)
            img = Image.open(fh).convert("RGB")
            emb = get_embedding(img)
            embeddings.append(emb)
            ids.append(f["id"])
            names.append(f["name"])
        except Exception as e:
            st.warning(f"Skipped {f.get('name','?')} : {e}")
    if len(embeddings) == 0:
        st.error("No images indexed. Check folder ID & service account access.")
        return np.array([]), [], []
    return np.vstack(embeddings), ids, names

index_embeddings, file_ids, file_names = build_index()

# ---------------------------
# UI - Upload and search
if index_embeddings.size == 0:
    st.stop()

st.write("Upload a product photo and the app will find visually similar images from All Photos (Drive).")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded:
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, caption="Query image", use_column_width=False, width=300)
    q_emb = get_embedding(query_img)
    sims = np.dot(index_embeddings, q_emb) / (np.linalg.norm(index_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-10)
    top_k = st.slider("Number of results", 1, 10, 5)
    best_idx = np.argsort(sims)[::-1][:top_k]
    st.subheader("Top Matches")
    cols = st.columns(min(top_k, 5))
    for i, idx in enumerate(best_idx):
        file_id = file_ids[idx]
        file_name = file_names[idx]
        url = f"https://drive.google.com/uc?id={file_id}"
        if i < len(cols):
            with cols[i]:
                st.image(url, caption=f"{file_name}\nscore={sims[idx]:.3f}", use_column_width=True)
        else:
            st.image(url, caption=f"{file_name}\nscore={sims[idx]:.3f}", use_column_width=True)
