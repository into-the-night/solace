from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from .api.routes.chat import router as chat_router
from .api.routes.analysis import router as analysis_router

app = FastAPI()

app.include_router(chat_router)
app.include_router(analysis_router)

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")