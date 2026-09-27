import dotenv
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
import os
from dotenv import load_dotenv 

load_dotenv()



def get_llm():

    return ChatMistralAI(model = "ministral-14b-2512", mistral_api_key= os.getenv("MISTRAL_API_KEY"), temperature=0.4)

def split_transcript(transcipt:str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap =  200
    )

    return splitter.split_text(transcipt)

def summmarize(transcript:str) ->str:
    llm = get_llm()

    template = ChatPromptTemplate.from_messages([
        ('system',"Summarize the portion of a  transcript activity"),
        ("human","{text}")
    ]
    )

    map_chain = template | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summarize = [map_chain.invoke({"text":chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summarize)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","You are an expert  summarizer. Combine these partial summaries into one final professional  summary in bullet points"),
            ("human","{text}")
        ]
    )

    combined_chain= (
        RunnablePassthrough()| RunnableLambda(lambda x:{"text":x}) | combined_prompt |llm |StrOutputParser()
    )

    return combined_chain.invoke(combined)

def generate_title(transcript:str)->str:

    llm = get_llm()

    template = ChatPromptTemplate.from_messages(
        [
            ("system",
                "Based on the  transcript, generate a short professional title "
                "(max 8 words). Only return the title, nothing else."),
            ("human","{text}")
        ]
    )

    title_chain = (
         RunnablePassthrough()| RunnableLambda(lambda x:{"text":x}) | template|llm| StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])