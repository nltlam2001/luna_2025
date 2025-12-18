from fastapi.responses import JSONResponse


def error_response(http_code: int, code: str, message: str):
    return JSONResponse(
        status_code=http_code,
        content={
            "status": "error",
            "error": {
                "code": code,
                "message": message
            }
        }
    )


# ----- Custom error shortcuts -----

def ERR_INVALID_FILE_FORMAT():
    return error_response(
        400,
        "INVALID_FILE_FORMAT",
        "File tải lên không phải định dạng .mha, .mhd hợp lệ hoặc thiếu series_uid."
    )


def ERR_UNAUTHORIZED():
    return error_response(
        401,
        "UNAUTHORIZED",
        "Request thiếu hoặc sai Authorization: Bearer <token>."
    )


def ERR_FORBIDDEN():
    return error_response(
        403,
        "FORBIDDEN",
        "Token hợp lệ nhưng đã bị khóa hoặc vượt quá giới hạn số lần gọi API."
    )


def ERR_NOT_FOUND():
    return error_response(
        404,
        "NOT_FOUND",
        "Endpoint không tồn tại hoặc Service Model đang không hoạt động."
    )


def ERR_PROCESSING_ERROR(message="Lỗi xử lý mô hình."):
    return error_response(
        422,
        "PROCESSING_ERROR",
        message
    )


def ERR_INTERNAL_SERVER_ERROR():
    return error_response(
        500,
        "INTERNAL_SERVER_ERROR",
        "Lỗi hệ thống nội tại không xác định."
    )


def ERR_GATEWAY_TIMEOUT():
    return error_response(
        504,
        "GATEWAY_TIMEOUT",
        "Thời gian xử lý vượt quá giới hạn 600 giây."
    )
