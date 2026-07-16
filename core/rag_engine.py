import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vectorstore import build_vector_store,load_vector_store,get_retriever
from operator import itemgetter # Because of this Now the chain expects a dictionary.


def get_llm():

    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key= os.getenv("MISTRAL_API_KEY"), temperature=0.2)


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs):
        formatted.append(
            f"Transcript Chunk {i+1}:\n{doc.page_content}"
        )

    return "\n\n".join(formatted)



def build_rag_chain(transcript:str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k = 8)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(

        [(
            "system",
            """
You are an expert AI Video Assistant.

You answer questions ONLY using the retrieved transcript/video  context.

Instructions:

> User can refer the transcipt with different words like video and meeting and so on.so accordingly you should consider the context.
• Carefully read ALL transcript chunks before answering.
• Information may be spread across multiple chunks.
• The transcript may describe concepts using different wording than the user's question.
• Infer meaning when appropriate instead of looking for exact keyword matches.
• If the transcript partially answers the question, provide the available information.
• Explain concepts clearly and professionally.
• user can refer the transcipt with different words like video and meeting and so on.
• Quote important statements when useful.
• Never invent facts not supported by the transcript.
• Only reply "I could not find this information in the transcript." if the retrieved context truly contains no relevant information.

Transcript Context:
--------------------
{context}
--------------------
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
    )

    #full LCEL Rag pipeline 

    rag_chain = (

        {"context" : itemgetter("question") | retriever | RunnableLambda(format_docs),
        #  "question": RunnablePassthrough()
          "question": itemgetter("question") ,
           "chat_history" :itemgetter("chat_history")
         }
         |prompt|llm|StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=8)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert AI Video Assistant.

You answer questions ONLY using the retrieved transcript context.

Instructions:

• Carefully read ALL transcript chunks before answering.
• Information may be spread across multiple chunks.
• The transcript may describe concepts using different wording than the user's question.
• Infer meaning when appropriate instead of looking for exact keyword matches.
• If the transcript partially answers the question, provide the available information.
• Explain concepts clearly and professionally.
• Quote important statements when useful.
• Never invent facts not supported by the transcript.
• Only reply "I could not find this information in the transcript." if the retrieved context truly contains no relevant information.

Transcript Context:
--------------------
{context}
--------------------
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context" : itemgetter("question") | retriever | RunnableLambda(format_docs),
        #  "question": RunnablePassthrough()
           "question": itemgetter("question") ,
           "chat_history" :itemgetter("chat_history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question, chat_history) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke({"question": question,"chat_history":chat_history})
    print(f"answer :{answer}")
    return answer