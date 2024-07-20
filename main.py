import asyncio
import fastapi
import fastapi_poe as fp
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import StreamingResponse
import json

import os

POE_API_KEY = os.getenv('POE_API_KEY')

app = fastapi.FastAPI()
# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域，在生产环境中应该指定具体的域
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict

# Replace with your actual Poe API key


@app.options("/v1/chat/completions")
async def options_chat_completions():
    # 处理预检请求
    return {"message": "OK"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 首先解析请求体
    body = await request.json()
    print(body)
    # 创建 ChatCompletionRequest 对象
    chat_request = ChatCompletionRequest(**body)
    
    def convert_role(role):
        return 'bot' if role == 'assistant' else role

    poe_messages = [
        fp.ProtocolMessage(
            role=convert_role(msg.role),
            content=msg.content
        ) for msg in chat_request.messages
    ]

   

    # 检查是否请求流式响应
    is_stream = body.get('stream', False)

    if is_stream:
        return StreamingResponse(stream_response(poe_messages, chat_request), media_type="text/event-stream")
    else:
        return await non_stream_response(poe_messages, chat_request)

async def stream_response(poe_messages, chat_request):
    # 直接使用传入的 model 作为 bot_name
    bot_name = chat_request.model
    full_response = ""
    async for partial in fp.get_bot_response(messages=poe_messages, bot_name=bot_name, api_key=POE_API_KEY):
        if hasattr(partial, 'text'):
            chunk = partial.text
        elif isinstance(partial, str):
            chunk = partial
        else:
            continue

        full_response += chunk
        yield construct_sse_event(chunk, chat_request.model, full_response)

    yield construct_sse_event("[DONE]", chat_request.model, full_response, done=True)

def construct_sse_event(chunk, model, full_response, done=False):
    if done:
        return "data: [DONE]\n\n"
    
    response = {
        "id": f"chatcmpl-{hash(full_response)}"[:20],
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": chunk
                },
                "finish_reason": None
            }
        ]
    }
    return f"data: {json.dumps(response)}\n\n"

async def non_stream_response(poe_messages, chat_request):
     # 直接使用传入的 model 作为 bot_name
    bot_name = chat_request.model
    full_response = ""
    print(poe_messages)
    async for partial in fp.get_bot_response(messages=poe_messages, bot_name=bot_name, api_key=POE_API_KEY):
        if hasattr(partial, 'text'):
            full_response += partial.text
        elif isinstance(partial, str):
            full_response += partial

    return ChatCompletionResponse(
        id="chatcmpl-" + str(hash(full_response))[:10],
        object="chat.completion",
        created=int(asyncio.get_event_loop().time()),
        model=chat_request.model,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response
                },
                "finish_reason": "stop"
            }
        ],
        usage={
            "prompt_tokens": sum(len(msg.content.split()) for msg in chat_request.messages),
            "completion_tokens": len(full_response.split()),
            "total_tokens": sum(len(msg.content.split()) for msg in chat_request.messages) + len(full_response.split())
        }
    )

@app.get("/test")
async def completions():
    return {"message": "Chat completions endpoint working"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)