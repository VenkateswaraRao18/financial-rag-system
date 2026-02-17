from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from rag_pipeline import RAGPipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()


class Query(BaseModel):
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # Ensure folder exists
    os.makedirs("uploaded_docs", exist_ok=True)

    # Clear previous uploaded files
    for old_file in os.listdir("uploaded_docs"):
        old_path = os.path.join("uploaded_docs", old_file)
        if os.path.isfile(old_path):
            os.remove(old_path)

    # Save new file
    file_path = os.path.join("uploaded_docs", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Rebuild index from only this file
    pipeline.build_index("uploaded_docs")

    return {"message": "File uploaded and indexed successfully"}


@app.post("/ask")
def ask_question(query: Query):
    answer = pipeline.ask(query.question)
    return {"answer": answer}
