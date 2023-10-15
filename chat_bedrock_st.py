import time

import boto3
import streamlit as st
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import traceback
import time
from typing import Any, Dict, List, Optional
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper


st.title("GenerativeAI: Optimizing Schedules Easily")

import json
import os
import sys

import boto3

module_path = "."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww


# ---- ⚠️ Un-comment and edit the below lines as needed for your AWS setup ⚠️ ----

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # E.g. "us-east-1"
# os.environ["AWS_PROFILE"] = "<YOUR_PROFILE>"
# os.environ["BEDROCK_ASSUME_ROLE"] = "<YOUR_ROLE_ARN>"  # E.g. "arn:aws:..."
os.environ["BEDROCK_ENDPOINT_URL"] = "https://bedrockForYoga"  # E.g. "https://..."


boto3_bedrock = bedrock.get_bedrock_client(
)

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

class BedrockEmbeddingsCustom(BedrockEmbeddings):
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a Bedrock model.

        Args:
            texts: The list of texts to embed

        Returns:
            List of embeddings, one for each text.
        """
        print(f"BedrockEmbeddingsCustom: embed_docs():: lenght of texts={len(texts)}::")
        results = []
        counter = 1
        errors = []
        for text in texts:
            try:
                response = self._embedding_func(text)
                results.append(response)
                #print(f"BedrockEmbeddingsCustom: embed_docs()::processed doc_{counter}:")
                counter+=1
            except:
                print(f"BedrockEmbeddingsCustom: ERROR ={traceback.format_exc()}:: WAITING for 20 SEC")
                time.sleep(20) # 20 sec
                errors.append(text)
        
        print(f"BedrockEmbeddingsCustom: embed_docs(): TRYING Errors now:len={len(errors)}:")
        for text in errors:
            print(f"BedrockEmbeddingsCustom: embed_docs(): error :text={text}:")
            try:
                response = self._embedding_func(text)
                results.append(response)
                #print(f"BedrockEmbeddingsCustom: embed_docs()::processed doc_{counter}:")
                counter+=1
            except:
                print(f"BedrockEmbeddingsCustom: ERROR ={text}:: WAITING for 20 SEC")
                time.sleep(20) # 20 sec
                    
        return results
    
bedrock_embeddings = BedrockEmbeddingsCustom(client=boto3_bedrock)
bedrock_embeddings

loader = PyPDFDirectoryLoader("./yoga_data/")

documents = loader.load()
# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
)
docs = text_splitter.split_documents(documents)

vectorstore_faiss = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

vectorstore_faiss.save_local("faiss_index")

vectorstore_faiss = FAISS.load_local("faiss_index", bedrock_embeddings)
wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

@st.cache_resource
def load_llm():
    # - create the Anthropic Model
    llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':2000, 'temperature':0})
    bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)

    prompt_template = """You will be acting as a Yoga Instructor. Use the follow [instructions] to answer the [question] based on the [context] provided. Don't answer if the answer is not present in the [context]. Follow the [output_format] given below while responding.

    context = {context} 

    instructions = Use following instructions to answer the question above. 
    Make sure to include following [attributes] in your answer as applicable.
    - Teach one yoga pose for the {question} asked.
    - Teach the benefit of the yoga pose for mental health
    - Teach the benefit of the yoga pose for physical health


    output_format = Provide your output as a text based paragraph that follows the instructions below
    - Each and every sentence is complete, ending with a full stop



    question = {question} 


    output_format = Provide your output as a detailed paragraph that contains all [attributes] following all [instructions] above

    answer: """


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm_titan,
        chain_type="stuff",
        retriever=vectorstore_faiss_titan.as_retriever(
            search_type="similarity", search_kwargs={"k": 9}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    #query = "What are the different ways that the medication lithium can be given?"
    query = "I am old and I have back pain in the lower back. How can I cure this using Yoga?"
    result = qa({"query": query})

    model = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

    return model


model = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What are you looking for today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # prompt = prompt_fixer(prompt)
        result = model.predict(input=prompt)

        # Simulate stream of response with milliseconds delay
        for chunk in result.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
