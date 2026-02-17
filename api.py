import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil

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


@app.get("/")
def health_check():
    return {"status": "running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    os.makedirs("uploaded_docs", exist_ok=True)

    for old_file in os.listdir("uploaded_docs"):
        old_path = os.path.join("uploaded_docs", old_file)
        if os.path.isfile(old_path):
            os.remove(old_path)

    file_path = os.path.join("uploaded_docs", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pipeline.build_index("uploaded_docs")

    return {"message": "File uploaded and indexed successfully"}


@app.post("/ask")
def ask_question(query: Query):
    answer = pipeline.ask(query.question)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
