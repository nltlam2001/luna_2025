from fastapi import Request
from fastapi.responses import JSONResponse
from utils.error_codes import ERR_UNAUTHORIZED

EXPECTED_TOKEN = "cac-van-de-hien-dai-khmt"

async def auth_middleware(request: Request, call_next):
    # Skip authentication for root path
    if request.url.path in ["/", "/docs", "/openapi.json"]:
        return await call_next(request)

    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        return ERR_UNAUTHORIZED()

    token = auth_header.split(" ")[1]
    if token != EXPECTED_TOKEN:
        return ERR_UNAUTHORIZED()

    return await call_next(request)
