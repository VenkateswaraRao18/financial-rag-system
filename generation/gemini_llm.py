import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3-flash-preview")



def generate_answer(context, question):
    prompt = f"""
You are a financial analyst assistant.
Answer ONLY using the context below.
If answer is not in context, say "Not found in report."

Context:
{context}

Question:
{question}
"""
    response = model.generate_content(prompt)
    return response.text


# What were the main revenue drivers in 2024?