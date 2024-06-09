Sure, here's a well-designed GitHub README markdown for your projects:

---

# RAGscribe

A chatbot application leveraging Google's Gemini LLM for document queries using Retrieval-Augmented Generation (RAG). Features include context preservation, handling out-of-context queries, and document summarization.

## Project Links

- **Spring Boot Server:** [Spring-RAG-Chatbot](https://github.com/AwadhootK/Spring-RAG-Chatbot)
- **React.js Frontend:** [RAG-Chatbot-Frontend](https://github.com/AwadhootK/RAG-Chatbot-Frontend)

## Technologies Used

- **Backend:** 
  - Spring Boot
  - FastAPI
  - AWS (S3, RDS)
  - PostgreSQL

- **Frontend:** 
  - React.js

- **Other:** 
  - Langchain
  - ChromaDB

## Features

- Integration with Google's Gemini LLM using Langchain and ChromaDB for:
  - Retrieval-Augmented Generation (RAG)
  - Document Summarization
  - Out-of-context Question Answering

- **Backend:**
  - Developed FastAPI server to expose LLM features to the Spring Boot server
  - Handles local and cloud file storage on AWS S3
  - Data manipulation using Spring Data JPA for PostgreSQL on AWS RDS

- **Frontend:**
  - Built using React.js
