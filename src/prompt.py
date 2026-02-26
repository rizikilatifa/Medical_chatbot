from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a medical assistant. Use only the provided context to answer.
If the answer is not in context, say you do not know.

<context>
{context}
</context>

Question: {input}
""")