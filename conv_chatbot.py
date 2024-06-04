import os
import warnings
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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


class RAGConversationalChatbot:

    def __init__(self, pdf_paths):
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=self.GOOGLE_API_KEY)

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro", temperature=0.7, convert_system_message_to_human=True)
        self.embeddings = CustomGoogleGenerativeAIEmbeddings(
            model="models/embedding-001")
        self.pdf_paths = pdf_paths

    def init_convo(self):
        vector_index = self.load_and_index_pdfs()
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, vector_index, contextualize_q_prompt)

        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)

        return rag_chain

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

    def answer(self, query):
        rag_chain = self.init_convo()
        chat_history = []
        try:
            response = rag_chain.invoke({
                'input': query,
                'chat_history': chat_history
            })

            chat_history.extend(
                [HumanMessage(content=query), response['answer']])
            return response['answer']
        except Exception:
            return 'Error'

    def chat(self):
        rag_chain = self.init_convo()

        chat_history = []
        while True:
            question = input("Ask a question (or type 'exit' to quit): ")
            if question.lower() == "exit":
                break

            try:
                response = rag_chain.invoke(
                    {"input": question, "chat_history": chat_history})
                chat_history.extend(
                    [HumanMessage(content=question), response["answer"]])
                print("Chatbot:", response["answer"])
                print('*'*50)

            except Exception as e:
                print(f"Error using Gemini API: {e}")
