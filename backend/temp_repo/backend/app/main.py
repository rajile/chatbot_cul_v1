from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .database import get_db, engine, Base
from .models import ChatRecord
from .bot_logic import CULBot
from pydantic import BaseModel
import os

# Crear tablas en la base de datos
Base.metadata.create_all(bind=engine)

app = FastAPI(title="CUL Chatbot API")

# Middleware para CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el bot (esto puede tardar, en producción se cargaría antes)
# En producción Railway se asume que el adaptador está en la carpeta /app/adapter
bot = CULBot(adapter_path=os.getenv("ADAPTER_PATH", "./adapter"))

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        # Generar respuesta
        response = bot.generate_response(request.message)
        
        # Guardar en base de datos
        new_record = ChatRecord(user_query=request.message, bot_response=response)
        db.add(new_record)
        db.commit()
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(ChatRecord).order_by(ChatRecord.timestamp.desc()).limit(50).all()

# Servir archivos estáticos del frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
