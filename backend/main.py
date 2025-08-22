from fastapi import FastAPI

app = FastAPI(title="chat-app", version="0.1")

@app.get("/api/health")
def health():
    return {"status": "ok"}
