from conv_chatbot import RAGConversationalChatbot

rcb = RAGConversationalChatbot(
    pdf_paths=['docs/Awadhoot Khutwad Resume (1).pdf']
)

rcb.load_and_index_pdfs()

print(rcb.answer("what did Ayush do at MIBAiO?"))
print(rcb.answer("what did Awadhoot do at MIBAiO?"))
