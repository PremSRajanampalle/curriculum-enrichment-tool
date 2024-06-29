import streamlit as st
    #from langchain.document_loaders import YoutubeLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
    #from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
    #from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
from bot_user_template import bot_template, user_template, css
from langchain_community.chat_models import ChatOpenAI
from readme import readme

#                                                   FOR VIDEO:
def summarize_video(video_url, client, chunk_size=10000, chunk_overlap=100, verbose=True):
    try:
        load = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        result = load.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(result)

        #chain = load_summarize_chain(llm=client, chain_type="map_reduce", verbose=verbose)# is llm == client
        #summary = chain.run(texts)

        template = """
        Given texts {texts}, your job is to build a comprehensive summary on that text... The summary should be between \n
        2000 to 3000 words long. 
        """
        response_summary = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": template.format(texts=texts)}],
            #max_tokens = 2000,
            temperature=0.0,
        )
        summary = response_summary.choices[0].message.content
        return summary

    except Exception as e:
        return f"Error: {e}"


def expand_summary(summary, num_of_tokens, client):
    expand_template = """
    Given a text, your job is to expand the text to the limit of {num_of_tokens}. But, when you do it, keep in mind to not \n
    alter the main idea behind the text. 

    Text: {summary}
    Number of tokens: {num_of_tokens}
    Expand: 
    """
    response_expanded_summary = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": expand_template.format(summary = summary, num_of_tokens = num_of_tokens)}],
        temperature=0.0,
    )

    expanded_response = response_expanded_summary.choices[0].message.content
    return expanded_response

#                                                           FOR BOOKS:
def search():
    input_text = st.text_input("Search the book you want")
    return input_text


# Function to generate book summary
def summary_func(input_text, client):
    template = """
        Given a book name {input_text}, it is your job to build a 2000 
        words of comprehensive study and analysis on it.The study should 
        start with the following things: Name of the book, author of the 
        book, date of publication, and table of contents of the book.
        Now, the analysis begins... for each chapter, for each topic 
        in that chapter, write a detailed 200 words study. Make sure 
        to complete for all the chapters. Dont miss any chapters in 
        between. Write chapter name in Bold, and write the analysis 
        of that in normal font structure. 
        """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": template.format(input_text=input_text)}],
        max_tokens=2000,
        temperature=0.7,
    )

    summary = response.choices[0].message.content
    return summary


def create_the_quiz_prompt_template(num_questions, quiz_type, quiz_context, client):
    """Create the prompt template for the quiz app."""

    template = """
You are an expert quiz maker for technical fields. Let's think step by step and
create a quiz with {num_questions} {quiz_type} questions about the following concept/content: {quiz_context}.

The format of the quiz could be one of the following:
Multiple-choice: 
Questions:
    <Question1>: <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
    <Question2>: <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
    ....
- Answers:
    <Answer1>: <a|b|c|d>
    <Answer2>: <a|b|c|d>
    ....
    Example:
    - Questions:
    - 1. What is the time complexity of a binary search tree?
        a. O(n)
        b. O(log n)
        c. O(n^2)
        d. O(1)
    - Answers: 
        1. b

The Multiple Choice Questions should not be like following: 

- Questions:
- 1. What Chapter talks about Linear Regression? 
    a. Chapter 1
    b. Chapter 2
    c. Chapter 3
    d. Chapter 4

Avoid the following type of Questions: 
 - What is the main focus of the chapter "Everything is an Object" in the book "Thinking In Java"? 
 - Which chapter in the book covers the use of interfaces to define contracts for classes?
 - What is the purpose of the chapter on "Holding Your Objects" in the book?
 - Which chapter in the book discusses the use of arrays in Java?

 Instead ask questions like following: 
 - What are Objects? 
 - What are interfaces? 
 - What do we mean by Holding Your Objects? 
 - What are arrays? 
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": template.format(num_questions=num_questions, quiz_type = quiz_type, quiz_context = quiz_context)}],
        max_tokens=2000,
        temperature=0.7,
    )

    response_quiz = response.choices[0].message.content
    return response_quiz

def split_questions_answers(quiz_response):
    """Function that splits the questions and answers from the quiz response."""
    questions = quiz_response.split("Answers:")[0]
    answers = quiz_response.split("Answers:")[1]
    return questions, answers

#                                                    FOR PDF CHATBOT:
def get_pdf_text(pdf_docs):
    text = ""  # an empty text on which we will append the processed texts
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # initializing an object - pdf reader to read pdf files
        for page in pdf_reader.pages:  # for every page in the pdf, append the text present in the pdf to the empy text above
            text += page.extract_text()
    return text


# Function to get text chunks from the PDF text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(  # create an object for creating text chunks -
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):  # for getting vectorestores
    embeddings = OpenAIEmbeddings()  # initialize openaiembeddings module
    vectorstore = FAISS.from_texts(texts=text_chunks,
                                   embedding=embeddings)  # use FAISS module to create the vectorestore
    return vectorstore


# Function to create a conversation chain
def get_conversation(vectorstore):
    llm = ChatOpenAI()  # initializing chat conversations
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # create memory
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),
                                                               memory=memory)  # create conversations
    return conversation_chain


# Function to handle user input for PDF chatbot
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    import os
    from openai import OpenAI

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override = True)

    st.set_page_config(page_title="Curriculum Tool", page_icon=":books:")
    st.title("Curriculum enrichment tool")

    # Sidebar for selecting different functionalities
    option = st.sidebar.selectbox("Select Option", ("README", "Book Summarization", "Video Summarization", "PDF Chatbot", "Quiz on Summarized Book", "Quiz on Summarized Video"))
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key', type='password')

    if option == "README":
        st.write(readme)

    # Book Summarization
    elif option == "Book Summarization":
        st.header("Book Summarization")
        if api_key:
            client = OpenAI(api_key= api_key)
            st.session_state.client = client

            input_text = search()
            if st.button("Generate Summary"):

                with st.spinner("Summarizing..."):
                    book_summary = summary_func(input_text, client)
                    st.session_state.book = book_summary
                st.success("Summary generated!")
                st.write(book_summary)

        else:
            st.write("Please provide api key...")

    elif option == "Quiz on Summarized Book":
        st.write("Quiz generator on THE BOOK")

        # Generate quiz
        num_questions = st.number_input("Enter the number of questions", min_value=1, max_value=10, value=3)
        quiz_type = st.selectbox("Select the quiz type", ["multiple-choice"])
        quiz_context = st.session_state.book
        response_quiz = create_the_quiz_prompt_template(num_questions, quiz_type, quiz_context, st.session_state.client)
        try:

            if st.button("Generate Quiz"):
                st.write("Quiz Generated!")
                questions, answers = split_questions_answers(response_quiz)
                st.session_state.answers = answers
                st.session_state.questions = questions
                st.write(questions)
            if st.button("Show Answers"):
                st.markdown(st.session_state.questions)
                st.write("----")
                st.markdown(st.session_state.answers)
        except:
            print("Please provide the name of the book. For that, go to 'Book Summary' section, and generate a book summary")

    elif option == "Video Summarization":
        st.header("Video Summarization")
        if api_key:
            client = OpenAI(api_key= api_key)
            st.session_state.client = client
            video_url = st.text_input("Enter YouTube video URL:")
            if 'button1_clicked' not in st.session_state:
                st.session_state.button1_clicked = False
                st.session_state.action1_performed = False
                st.session_state.summary = None  # Initialize summary in session state

                # Button 1 with action
            if not st.session_state.button1_clicked:
                if st.button("Get Transcript"):
                    # Perform action 1
                    with st.spinner("Transcribing..."):
                        summary = summarize_video(video_url, st.session_state.client)
                    st.success("Transcript generated!")
                    st.write(summary)

                    st.session_state.button1_clicked = True
                    st.session_state.action1_performed = True  # Track completion of action 1
                    st.session_state.summary = summary  # Store summary in session state
            else:
                st.write(st.session_state.summary)

            with st.form("user_inputs"):
                num_of_tokens = st.number_input("How many words?", max_value=2000)
                expand_button = st.form_submit_button("Summary")

                if expand_button:  # Now check if the button was actually clicked
                    if st.session_state.action1_performed and num_of_tokens is not None:
                        if st.session_state.summary is None:
                            st.error("Please generate a transcript first.")
                        else:
                            with st.spinner("Summarizing..."):
                                summary_new = expand_summary(st.session_state.summary, num_of_tokens, st.session_state.client)
                                st.session_state.new_sum = summary_new
                            st.write("**Summary**")
                            st.write(summary_new)
                            # Perform action 2
                            st.write("Action 2 performed!")

    elif option == "Quiz on Summarized Video":
        st.write("Quiz generator on THE VIDEO")
        # Generate quiz
        num_questions = st.number_input("Enter the number of questions", min_value=1, max_value=10, value=3)
        quiz_type = st.selectbox("Select the quiz type", ["multiple-choice"])
        quiz_context = st.session_state.new_sum
        response_quiz = create_the_quiz_prompt_template(num_questions, quiz_type, quiz_context, st.session_state.client)
        try:

            if st.button("Generate Quiz"):
                st.write("Quiz Generated!")
                questions, answers = split_questions_answers(response_quiz)
                st.session_state.answers = answers
                st.session_state.questions = questions
                st.write(questions)
            if st.button("Show Answers"):
                st.markdown(st.session_state.questions)
                st.write("----")
                st.markdown(st.session_state.answers)
        except:
            print("Please provide the name of the book. For that, go to 'Book Summary' section, and generate a book summary")

# PDF Chatbot
    else:
        load_dotenv()
        st.header("PDF Chatbot")
        st.write(css, unsafe_allow_html=True)
        if "conversation" not in st.session_state:
            st.session_state.conversation = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        pdf_docs = st.file_uploader("Upload PDFs here and click 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation(vectorstore)
        user_question = st.text_input("Prompt here:")
        if user_question:
            handle_user_input(user_question)

main()