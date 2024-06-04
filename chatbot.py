import os
import warnings
from dotenv import load_dotenv
from typing import List, Optional

import google.generativeai as genai
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
warnings.filterwarnings("ignore")
load_dotenv('.env')


class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str], task_type: Optional[str] = None, titles: Optional[List[str]] = None, output_dimensionality: Optional[int] = None) -> List[List[float]]:
        embeddings_repeated = super().embed_documents(
            texts, task_type, titles, output_dimensionality)
        embeddings = [list(emb) for emb in embeddings_repeated]
        return embeddings

    def as_retriever(self, search_kwargs):
        super().as_retriever(search_kwargs=search_kwargs)


class RAGChatbot:

    def __init__(self, pdf_paths):
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=self.GOOGLE_API_KEY)

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro", temperature=0.7, convert_system_message_to_human=True)
        self.embeddings = CustomGoogleGenerativeAIEmbeddings(
            model="models/embedding-001")
        self.pdf_paths = pdf_paths

    def load_and_index_pdfs(self):
        pages = []
        texts = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=1000)

        for pdf_path in self.pdf_paths:
            pdf_loader = PyPDFLoader(pdf_path)
            new_pages = pdf_loader.load_and_split()
            pages.extend(new_pages)
            context = "\n\n".join(str(p.page_content) for p in new_pages)
            texts.extend(text_splitter.split_text(context))

        vector_index = Chroma.from_texts(
            texts, self.embeddings).as_retriever(search_kwargs={"k": 5})
        return vector_index

    def init_template(self):
        vector_index = self.load_and_index_pdfs()

        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "\nthanks for asking!" at the end of the answer.
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)  # Run chain
        self.qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=vector_index,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

    def chat(self):
        self.init_template()

        while True:
            question = input("Ask a question (or type 'exit' to quit): ")
            if question.lower() == "exit":
                break

            try:
                result = self.qa_chain({"query": question})
                print("Generated answer from Gemini API:", result["result"])

            except Exception as e:
                print(f"Error using Gemini API: {e}")
                
