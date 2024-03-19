from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from .lcel_version import chain
from langserve import add_routes

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

print(chain.InputType)

# Edit this to add the chain you want to add
add_routes(app, chain, path='/qa', playground_type="chat")

if __name__ == "__main__":
    from dotenv import load_dotenv
    import uvicorn
    load_dotenv()

    uvicorn.run(app, host="0.0.0.0", port=8000)
