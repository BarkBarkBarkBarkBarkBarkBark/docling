import os
import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAI
import matplotlib.pyplot as plt
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory, RedisChatMessageHistory
import re  # For filtering generated code blocks

# Configure API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set. Please check your environment variables.")
    st.stop()

client = OpenAI(api_key=api_key)

# Agent settings
SQL_AGENT_VERBOSE = os.getenv("SQL_AGENT_VERBOSE")

# Configuration for memory storage: Redis or in-memory
USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Function to retrieve the chat history
def get_chat_history(session_id: str):
    if USE_REDIS:
        return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = ChatMessageHistory()
        return st.session_state.chat_history

# Initialize the database and the LLM model
db = SQLDatabase.from_uri("sqlite:///export.db")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Prompt Template for the SQL agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert SQL querying assistant analyzing expense data from a bank statement. "
         "The user will ask questions about expenses, and your task is to query a SQL database to find relevant answers. "
         "The database contains a table named 'data' with a 'description' column that stores transaction details.\n\n"
         "Guidelines:\n"
         "- **Interpret the descriptions**: Bank statement descriptions may not be clear or consistent. "
         "You must infer the meaning based on keywords or partial matches.\n"
         "- Avoid bias: Do not assume specific vendors, stores, or categories unless explicitly mentioned by the user.\n"
         "- Query the database to find transactions that match the user's request as closely as possible.\n"
         "- If you cannot find exact results, suggest alternative keywords or clarify the user's question."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
        ("user", "{agent_scratchpad}")
    ]
)

# Create the SQL agent
sql_agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    prompt=prompt,
    max_iterations=30,
    verbose=SQL_AGENT_VERBOSE
)

# Function to filter and extract Python code block
def extract_code_block(response):
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if code_match:
        return code_match.group(1)  # Return only the code block
    return None

# Function to generate a chart using OpenAI API
def generate_chart_code(data, user_request):
    prompt = f"Generate a Python Matplotlib script to visualize the following data:\n{data}\n\n{user_request}."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that generates Python Matplotlib scripts. DO NOT INCLUDE ANY EXPLANATION JUST PYTHON CODE"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Streamlit UI setup
st.title("CreditStatsBot")

if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(8).hex()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Retrieve chat history
chat_history = get_chat_history(session_id=st.session_state.session_id)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_query := st.chat_input("Sugar and Credit miner, ask me about your expenses"):
    chat_history.add_user_message(user_query)
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Looking for sugar in our database..."):
        try:
            # Execute the SQL agent
            response = sql_agent.invoke({"input": user_query, "history": chat_history.messages})
            formatted_response = response.get("output", "No result found.")

# Check for chart request
            if "chart" in user_query.lower() or "graph" in user_query.lower():
                chart_code = generate_chart_code(formatted_response, user_query)
                cleaned_chart_code = extract_code_block(chart_code)
            
                if cleaned_chart_code:
                    # Collapsable section for generated code
                    with st.expander("Generated Chart Code", expanded=False):
                        st.code(cleaned_chart_code, language="python")
            
                    # Render the chart
                    try:
                        exec(cleaned_chart_code, {"plt": plt})
                        st.pyplot(plt.gcf())  # Render the current figure
                    except Exception as chart_error:
                        st.error(f"Failed to render chart: {chart_error}")
                else:
                    st.error("No valid Python code block was found in the response.")
            else:
                with st.chat_message("assistant"):
                    st.markdown(formatted_response)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})

            # Add the AI's response to the chat history
            chat_history.add_ai_message(formatted_response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")