import asyncio
import os
import shutil

import boto3
from fastapi import (BackgroundTasks, FastAPI, File, Form, HTTPException,
                     UploadFile)
from pydantic import BaseModel

# from chatbot import RAGChatbot
from conv_chatbot import RAGConversationalChatbot


class Query(BaseModel):
    query: str


app = FastAPI()


class Data:

    def __init__(self):
        self.pdf_paths = Data.get_all_files("docs")
        self.rcb = RAGConversationalChatbot(pdf_paths=self.pdf_paths)

    @staticmethod
    def get_all_files(directory):
        files = []

        if not os.path.exists("docs"):
            os.makedirs("docs")
        for entry in os.listdir(directory):
            full_path = os.path.join(directory, entry)
            if os.path.isfile(full_path):
                files.append(full_path)
        return files

    @staticmethod
    def empty_folder(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                Data.empty_folder(item_path)
                os.rmdir(item_path)

    @staticmethod
    def upload_file_to_s3(file, name):
        S3_BUCKET = 'awadhoot-rag-chatbot-files'
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(S3_BUCKET)
        bucket.put_object(Key=name, Body=file)

    @staticmethod
    def check_and_clean_folder(folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        else:
            os.makedirs(folder_path)

    @staticmethod
    def download_files_from_s3(username):
        s3 = boto3.resource('s3')
        bucket_name = 'awadhoot-rag-chatbot-files'
        bucket = s3.Bucket(bucket_name)
        s3_folder = username
        local_dir = None

        Data.check_and_clean_folder("docs")

        for obj in bucket.objects.filter(Prefix=s3_folder):
            target = obj.key if local_dir is None \
                else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == '/':
                continue
            bucket.download_file(obj.key, target)

        os.rename(s3_folder, "docs")
        # restore()
        return "Done"


data = Data()


def restore():
    global data
    data.rcb = RAGConversationalChatbot(pdf_paths=data.get_all_files("docs"))
    data.rcb.load_and_index_pdfs()


@app.get("/ping")
async def ping():
    return "pong"


@app.post("/ask")
async def respond(query: Query):
    global data
    ans = data.rcb.answer(query.query)
    return {'query': query.query, 'answer': ans}


@app.get("/empty-context")
async def empty():
    global data
    data.empty_folder('docs')
    data.rcb.clear()
    return "done"


@app.post("/upload")
async def upload_file(
    index: bool = Form(...),
    save: bool = Form(...),
    username: str = Form(...),
    file: UploadFile = File(...)
):
    if not file:
        raise HTTPException(
            status_code=400,
            detail="File not found"
        )

    print(f'index={index}, save={save}, username={username}')
    file_location = f"docs/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    global data
    if index:
        data.rcb = RAGConversationalChatbot(
            pdf_paths=data.get_all_files("docs"))
    if save:
        contents = file.file.read()
        Data.upload_file_to_s3(
            file=contents, name=f'{username}/{file.filename}')
        pass
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}


@app.post("/download-context-files")
async def restore_user_files(username: str = Form(...)):
    result = Data.download_files_from_s3(username=username)

    if result == 'Done':
        return "Context successfully restored"
    

@app.post("/answer-llm")
async def answer_llm(query: Query):
    global data
    response = data.rcb.answer_from_llm(query=query.query)
    return {'query': query, 'response': response}


@app.get("/summarize")
async def summarize_doc():
    global data
    data.rcb.load_and_index_pdfs()
    if data.rcb.texts == None or data.rcb.texts == []:
        return {'summary': 'No document to summarize'}
    summary = data.rcb.summarize_from_llm(data.rcb.texts)
    return {'summary': summary}


@app.post("/semantic_search")
async def semantic_search(query: Query):
    global data
    semantic_result = data.rcb.sematic_doc_search_by_vector(query=query.query)
    return {'similar_doc': semantic_result}

# app.run(host='0.0.0.0', port=5000)
