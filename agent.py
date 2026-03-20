from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

def fraud_agent(transaction_data, model_result):

    prompt = f"""
You are a bank fraud analyst.

Transaction: {transaction_data}
Model Result: {model_result}

Give a SHORT explanation (max 3 lines):
- fraud reason
- risk factor
- action
"""

    return llm.invoke(prompt)