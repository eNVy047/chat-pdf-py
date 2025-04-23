# MultiPDF Chat App
## Introduction
------------
The MultiPDF Chat App is a Python application that allows you to chat with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.

## How It Works
------------

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

The application follows these steps to provide responses to your questions:

1. PDF Loading: The app reads multiple PDF documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

## Dependencies and Installation
----------------------------
To install the MultiPDF Chat App, please follow these steps:

1. Clone the repository to your local machine.
2. ```
   git clone https://github.com/eNVy047/chat-pdf-py.git
   ```

    ```
    cd chat-pdf-py
   ```

3. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

4. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
    #Already added.

## Usage
-----
To use the MultiPDF Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded PDFs using the chat interface.


### 📫 Connect with me

<p align="left">
  <a href="mailto:narayan7154@gmail.com">
    <img align="center" src="https://img.shields.io/badge/-Email-D14836?logo=gmail&logoColor=white&style=flat" />
  </a>
  <a href="https://www.linkedin.com/in/narayanverma/" target="_blank">
    <img align="center" src="https://img.shields.io/badge/-LinkedIn-0077B5?logo=linkedin&style=flat" />
  </a>
  <a href="https://www.instagram.com/narayan_.v/" target="_blank">
    <img align="center" src="https://img.shields.io/badge/-Instagram-E4405F?logo=instagram&logoColor=white&style=flat" />
  </a>
  <a href="https://narayanverma.vercel.app" target="_blank">
    <img align="center" src="https://img.shields.io/badge/-Portfolio-24292E?logo=githubpages&style=flat" />
  </a>
</p>

### ☕ Support Me

If you like my work and want to support me, you can buy me a coffee!  
[![Buy Me a Coffee](https://img.shields.io/badge/-Buy%20me%20a%20coffee-FFDD00?logo=buy-me-a-coffee&logoColor=black&style=flat)](https://www.buymeacoffee.com/narayanverma)

### 📄 License
This project is open-source. You’re free to use, share, and modify it for personal and commercial projects.
Let me know if you’d like me to generate a `CONTRIBUTING.md` or `vercel.json` for deployment next!


## Contributing
------------
This repository is intended for educational purposes and does not accept further contributions. Feel free to utilize and enhance the app based on your own requirements.
