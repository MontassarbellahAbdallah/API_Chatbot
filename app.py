from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Configure the Generative AI model
model = genai.GenerativeModel("gemini-pro") 
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Initialize the Streamlit app
st.set_page_config(page_title="Gemini Application")

st.header("Ask with Gemini")

# Initialize session state for chat history if it doesn't exist
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Sidebar with expandable history items
st.sidebar.header("Chat History")

def delete_history(index):
    del st.session_state.history[index]

# Display history in the sidebar with expanders
for i, entry in enumerate(st.session_state.history):
    with st.sidebar.expander(f"Q{i+1}"):
        st.write(f"**Input:** {entry['input']}")
        st.write(f"**Response:** {entry['response']}")
        if st.button(f"Delete", key=f"delete_{i}"):
            delete_history(i)
            # Force rerun by setting query parameters
            st.query_params.update(deleted_index=i)

# Input and submit button for new questions
input = st.text_input("", key="input")
submit = st.button("Submit")



if submit and input:
    response = get_gemini_response(input)
    response_text = ''.join([chunk.text for chunk in response])
    st.session_state.history.append({"input": input, "response": response_text})
    st.write(response_text)
    st.query_params.update(new_question=input)

# streamlit run app.py