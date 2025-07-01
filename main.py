from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

model = OllamaLLM(model="phi:latest")

template = """
Your are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here are some question reviews: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    reviews = retriever.invoke(question)
    result=chain.invoke({"reviews":reviews,"question": question})
    print(result)
