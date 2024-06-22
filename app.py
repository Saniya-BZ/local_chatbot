
###### test PERFECT

from flask import Flask, request, jsonify, send_from_directory
import os
from rag import ChatAI


app = Flask(__name__)
assistant = ChatAI()

DOCUMENTS_FOLDER = 'C:\\Users\\Administrator\\Downloads\\all_documents'
TEMPLATES_FOLDER = 'templates'

# Ensure documents folder exists
if not os.path.exists(DOCUMENTS_FOLDER):
    os.makedirs(DOCUMENTS_FOLDER)

@app.route('/ingest', methods=['POST'])
def ingest():
    files = request.files.getlist('files')
    for file in files:
        filename = file.filename
        file_path = os.path.join(DOCUMENTS_FOLDER, filename)
        file.save(file_path)
        if os.path.exists(file_path):
            print(f"Ingesting file from path: {file_path}")
            assistant.ingest(file_path)
        else:
            print(f"File path does not exist: {file_path}")
    return jsonify({"status": "success"})

@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query')
    # Re-initialize the assistant here if necessary
    if not assistant.chain:
        initialize_assistant()
    response = assistant.ask(query)
    return jsonify({"response": response})

@app.route('/fetch', methods=['GET'])
def fetch():
    documents = []
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith('.pdf'):
            documents.append(filename)
    return jsonify({"documents": documents})

@app.route('/')
def index():
    return send_from_directory(TEMPLATES_FOLDER, 'index.html')

def initialize_assistant():
    # Ingest all existing documents in the DOCUMENTS_FOLDER on startup
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith('.pdf'):
            file_path = os.path.join(DOCUMENTS_FOLDER, filename)
            if os.path.exists(file_path):
                print(f"Loading document on startup from path: {file_path}")
                assistant.ingest(file_path)
            else:
                print(f"File path does not exist: {file_path}")


if __name__ == '__main__':
    initialize_assistant()
    app.run(debug=True)


# from waitress import serve
# from app import app
 # serve(app, host='0.0.0.0', port=5000)
    




