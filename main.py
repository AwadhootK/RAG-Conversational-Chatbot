import os

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from chatbot import RAGChatbot
from conv_chatbot import RAGConversationalChatbot


class Query(BaseModel):
    query: str


app = FastAPI()


pdf_paths = ["docs/Internship Research Paper.pdf"]
rcb = RAGConversationalChatbot(pdf_paths=pdf_paths)


def get_all_files(directory):
    files = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files


def empty_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            empty_folder(item_path)
            os.rmdir(item_path)


@app.get("/ping")
async def ping():
    return "pong"


@app.post("/ask")
async def respond(query: Query):
    ans = rcb.answer(query.query)
    return {'query': query.query, 'answer': ans}


@app.post("/empty-context")
async def empty():
    empty_folder('docs')
    return "done"


@app.post("/upload/{index}")
async def upload_file(index: str, file: UploadFile = File(...)):
    file_location = f"docs/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    if index == 'yes':
        global rcb
        rcb = RAGConversationalChatbot(pdf_paths=get_all_files("docs"))
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}


# app.run(host='0.0.0.0', port=5000)
