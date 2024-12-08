from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
# Load environment variables
load_dotenv()

# Configure the Generative AI model
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("API key for Generative AI not found in environment variables.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define request and response models
class QuestionRequest(BaseModel):
    question: str

class ResponseModel(BaseModel):
    input: str
    response: str

# Store chat history (in-memory storage for simplicity; replace with DB for persistence)
chat_history = []
@app.post("/chat", response_model=ResponseModel)
async def chat_with_bot(request: QuestionRequest):
    """Répondre à une question envoyée en POST."""
    try:
        # Envoyer la question au modèle Gemini
        response = chat.send_message(request.question, stream=True)
        response_text = ''.join([chunk.text for chunk in response])

        # Sauvegarder dans l'historique
        chat_entry = {"response": response_text}
        chat_history.append(chat_entry)

        return ResponseModel(input=request.question, response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the question: {str(e)}")

@app.get("/ask/{question}", response_model=ResponseModel)
async def ask_question_path(question: str):
    """Answer a question passed in the URL path."""
    try:
        # Envoyer la question au modèle Gemini
        response = chat.send_message(question, stream=True)
        response_text = ''.join([chunk.text for chunk in response])

        # Sauvegarder dans l'historique
        chat_entry = {"response": response_text}
        chat_history.append(chat_entry)

        return ResponseModel(input=question, response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the question: {str(e)}")


@app.get("/history")
async def get_chat_history():
    """Retrieve the chat history."""
    return {"history": chat_history}

@app.delete("/history/{index}")
async def delete_history(index: int):
    """Delete a specific chat history entry by index."""
    try:
        if 0 <= index < len(chat_history):
            del chat_history[index]
            return {"message": "History entry deleted successfully."}
        else:
            raise IndexError("Invalid index.")
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))
