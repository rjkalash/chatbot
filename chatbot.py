import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai.chat_models import ChatOpenAI



OPENAI_API_KEY = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
st.header("AskMyPDF")
st.write("AskMyPDF is your intelligent document assistant that turns static PDFs into interactive knowledge hubs. Simply upload a PDF file, and AskMyPDF will analyze its content, allowing you to ask questions and get instant, accurate answers. Whether it’s research papers, manuals, contracts, or e-books, AskMyPDF makes understanding complex documents effortless and efficient.\n With advanced AI capabilities, AskMyPDF streamlines your workflow by delivering precise insights without the need to read through pages of text. It's perfect for students, professionals, and anyone seeking quick answers from their documents!")

#upload pdf file
with st.sidebar:
    st.title("Your Documents")
    file= st.file_uploader("Upload a PDF file and start asking questions", type='pdf')

#extract the text   
if file is not None:
    pdf_reader=PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        # st.write(text)

#breaking text into chunks
    text_splitter= RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000, 
        chunk_overlap=150, 
        length_function=len
    )

    chunks=text_splitter.split_text(text)
    # st.write(chunks)

    embeddings= OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_store =FAISS.from_texts(chunks, embeddings)

    user_question= st.text_input("Enter your question: ")

    if user_question:
        match=vector_store.similarity_search(user_question)
        # st.write(match)

        llm = ChatOpenAI(api_key=OPENAI_API_KEY,temperature=0, model="gpt-3.5-turbo", max_tokens=1000)

        chain= load_qa_chain(llm, chain_type="stuff")
        response= chain.run(input_documents= match, question= user_question)
        st.write(response)

    
