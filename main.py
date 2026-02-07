import os, json, uuid, jwt, datetime
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from llama_cpp import Llama
import whisper, pyttsx3, requests
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib .styles import getSampleStyleSheet

# ================= CONFIG =================
SECRET_KEY = "LERA_SUPER_SECRET"
ALGO = "HS256"

MODEL_PATH = "models/Phi-3-mini-4k-instruct-q4.gguf"  # <--- Güncel model yolu
DATA = "data"
USERS = f"{DATA}/users.json"
MEMORY = f"{DATA}/memory.json"
os.makedirs(DATA, exist_ok=True)

for f, d in [(USERS, {}), (MEMORY, [])]:
    if not os.path.exists(f):
        json.dump(d, open(f, "w"))

# ================= APP =================
app = FastAPI(title="Lera AI")
security = HTTPBearer()

llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)
whisper_model = whisper.load_model("base")
tts = pyttsx3.init()

# ================= HELPERS =================
def load(p): return json.load(open(p))
def save(p, d): json.dump(d, open(p,"w"), indent=2)

def token(data):
    return jwt.encode(
        {**data, "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)},
        SECRET_KEY,
        algorithm=ALGO
    )

def auth(creds: HTTPAuthorizationCredentials = Depends(security)):
    try:
        return jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGO])
    except:
        raise HTTPException(401, "Invalid token")

def web_search(q):
    r = requests.get("https://api.duckduckgo.com/", params={
        "q": q, "format": "json", "no_html": 1
    })
    return r.json().get("AbstractText","")

def speak(text):
    path = f"{DATA}/{uuid.uuid4()}.mp3"
    tts.save_to_file(text, path)
    tts.runAndWait()
    return path

def reply_llm(prompt):
    return llm(prompt, max_tokens=300)["choices"][0]["text"]

# ================= MODELS =================
class Login(BaseModel):
    username: str
    password: str

class Chat(BaseModel):
    message: str

# ================= AUTH =================
@app.post("/register")
def register(u: Login):
    users = load(USERS)
    if u.username in users:
        raise HTTPException(400,"User exists")
    users[u.username] = {"password": u.password}
    save(USERS, users)
    return {"ok": True}

@app.post("/login")
def login(u: Login):
    users = load(USERS)
    if users.get(u.username,{}).get("password") != u.password:
        raise HTTPException(401,"Wrong credentials")
    return {"token": token({"user": u.username})}

# ================= CHAT =================
@app.post("/chat")
def chat(req: Chat, user=Depends(auth)):
    mem = load(MEMORY)
    web = web_search(req.message) if any(x in req.message.lower() for x in ["bugün","güncel"]) else ""
    prompt = f"""
    Sen Lera'sın.
    Kullanıcı: {user['user']}
    İnternet: {web}
    Önceki konuşma: {mem[-3:]}
    Soru: {req.message}
    """
    reply = reply_llm(prompt)
    mem.append({"u":user["user"],"q":req.message,"a":reply})
    save(MEMORY, mem)
    return {"reply": reply, "audio": speak(reply)}

# ================= VOICE =================
@app.post("/voice")
async def voice(file: UploadFile, user=Depends(auth)):
    path = f"{DATA}/{uuid.uuid4()}.wav"
    open(path,"wb").write(await file.read())
    text = whisper_model.transcribe(path, language="tr")["text"]
    return chat(Chat(message=text), user)

# ================= PDF =================
@app.post("/math-pdf")
def math_pdf(topic: str, user=Depends(auth)):
    text = reply_llm(f"{topic} için detaylı matematiksel ispat yaz.")
    file = f"{DATA}/{uuid.uuid4()}.pdf"
    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()
    doc.build([Paragraph(p, styles["Normal"]) for p in text.split("\n")])
    return {"pdf": file}  
