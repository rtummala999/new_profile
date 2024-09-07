import os
import json
from flask import Flask, request, render_template, jsonify, redirect, url_for, session
import openai
from dotenv import load_dotenv
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as PineconeRetriever
# Load environment variables
import pinecone
from pinecone import Pinecone, ServerlessSpec
load_dotenv()
# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')

#Pinecone
cloud = 'aws'
region = 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)
api_key = os.environ.get("PINECONE_API_KEY")

# configure client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = 'gpt-4-langchain-docs-fast'

# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='cosine',
        spec=spec
    )

# connect to index
index = pc.Index(index_name)


# Load OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load documents and initialize the Pinecone index
def load_documents_and_initialize_db():
    try:
        # Path to the directory containing the documents
        docs_path = 'R:\\trash\\djano\\final_profile\\docs'
        documents = []

        # Iterate over all PDF files in the directory
        for filename in os.listdir(docs_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(docs_path, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings()

        # Create embeddings for the document chunks
        embedded_docs = embeddings.embed_documents([doc.page_content for doc in docs])

        # Create unique IDs for the document chunks
        ids = [str(i) for i in range(len(embedded_docs))]

        # Upsert the embeddings along with the document metadata
        metadata = [{'text': doc.page_content} for doc in docs]

        # Upsert the embeddings into Pinecone
        index.upsert(vectors=zip(ids, embedded_docs, metadata), namespace="pdf_docs1")

        return index

    except Exception as e:
        print(f"An error occurred while loading documents: {e}")
        return None


# Initialize Pinecone index
index = load_documents_and_initialize_db()

# Check if the index was successfully initialized
if index is None:
    raise ValueError("Failed to initialize the Pinecone index. Please check the document path and format.")


# Function to save the last three questions in the session
def save_question_in_session(question):
    if 'questions' not in session:
        session['questions'] = []

    # Append the new question
    session['questions'].append(question)

    # Keep only the last 3 questions
    session['questions'] = session['questions'][-3:]


# Save conversation to a file (Optional)
CONVERSATION_FILE = 'conversations.json'


def save_conversation(question, answer):
    try:
        if os.path.exists(CONVERSATION_FILE):
            with open(CONVERSATION_FILE, 'r') as f:
                conversations = json.load(f)
        else:
            conversations = []

        conversations.append({"question": question, "answer": answer})

        with open(CONVERSATION_FILE, 'w') as f:
            json.dump(conversations, f, indent=4)
    except Exception as e:
        print(f"Failed to save conversation: {e}")


# Route for index page
@app.route('/')
def index():
    return render_template('index.html')


# Route for handling user questions
@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['question']
    index_name = 'gpt-4-langchain-docs-fast'
    index = pc.Index(index_name)

    # Save the new question in the session
    save_question_in_session(query)

    try:
        # Retrieve the last 3 questions from the session
        previous_questions = session.get('questions', [])
        previous_questions_text = "\n".join(previous_questions)

        # Embed the query
        embeddings = OpenAIEmbeddings()
        query_embedding = embeddings.embed_query(query)

        # Query Pinecone for the most relevant documents
        response = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        # Extract the matching documents
        contexts = [match['metadata']['text'] for match in response['matches']]
        augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

        # System message to 'prime' the model
        primer = f"""You are a highly intelligent question-answering bot.
                     These are the previous questions asked in this session:
                     {previous_questions_text}

                     Answer the current question based on the provided context. If the answer cannot
                     be found in the context, respond with "I don't know and make sure you give straight answer within 2 lines ".
                     """

        # Use OpenAI's ChatCompletion (GPT-4) to generate an answer
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": augmented_query}
            ]
        )

        # Get the answer
        answer = response['choices'][0]['message']['content']

        # Save the conversation (optional)
        save_conversation(query, answer)

        return jsonify({'answer': answer})

    except Exception as e:
        answer = f"An error occurred: {e}"
        return jsonify({'answer': answer})


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
