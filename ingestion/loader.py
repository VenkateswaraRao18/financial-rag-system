from pypdf import PdfReader
import os


def load_pdfs(data_path="data"):
    documents = []

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(data_path, file))
            text = ""

            for page in reader.pages:
                text += page.extract_text() + "\n"

            documents.append({
                "file_name": file,
                "content": text
            })

    return documents
