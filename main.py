import os
import asyncio
import json
import base64
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Import our chatbot
from chatbot.multimodal_realtime import RealtimeChatbot

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview")
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "alloy")

# FastAPI app
app = FastAPI(
    title="Poppins - Realtime Parent Assistant",
    description="A realtime multimodal chatbot API for supporting parents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TextMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class AudioMessage(BaseModel):
    audio_data: str  # Base64 encoded audio
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    audio_data: Optional[str] = None  # Base64 encoded audio response

class SessionStats(BaseModel):
    session_id: str
    total_messages: int
    conversation_pairs: int
    user_messages: int
    assistant_messages: int

# Global chatbot sessions
chatbot_sessions: Dict[str, RealtimeChatbot] = {}

async def get_or_create_chatbot(session_id: str) -> RealtimeChatbot:
    """Get existing chatbot session or create new one"""
    if session_id not in chatbot_sessions:
        chatbot = RealtimeChatbot(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            voice=OPENAI_VOICE
        )
        chatbot_sessions[session_id] = chatbot
    return chatbot_sessions[session_id]

async def cleanup_chatbot(session_id: str):
    """Clean up chatbot session"""
    if session_id in chatbot_sessions:
        chatbot = chatbot_sessions[session_id]
        if chatbot.is_connected:
            await chatbot.disconnect()
        del chatbot_sessions[session_id]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "poppins-api"}

# Debug endpoint to test chatbot connection
@app.get("/debug/test-connection")
async def test_connection():
    """Test chatbot connection and basic functionality"""
    try:
        chatbot = RealtimeChatbot(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            voice=OPENAI_VOICE
        )
        
        # Test connection
        await chatbot.connect(modalities=["text"])
        
        # Test basic message
        response_text = ""
        response_complete = False
        
        def text_handler(chunk: str):
            nonlocal response_text
            response_text += chunk
        
        def complete_handler():
            nonlocal response_complete
            response_complete = True
        
        chatbot.set_text_handler(text_handler)
        chatbot.set_response_complete_handler(complete_handler)
        
        await chatbot.send_text_message("Hello, this is a test message.", include_audio=False)
        await chatbot.listen_for_responses()
        
        # Wait for completion
        timeout_counter = 0
        while not response_complete and timeout_counter < 50:
            await asyncio.sleep(0.1)
            timeout_counter += 1
        
        await chatbot.disconnect()
        
        return {
            "connection_successful": True,
            "response_received": len(response_text) > 0,
            "response_length": len(response_text),
            "response_preview": response_text[:100] if response_text else "",
            "timeout_reached": timeout_counter >= 50
        }
        
    except Exception as e:
        return {
            "connection_successful": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

# Text chat endpoint (non-streaming)
@app.post("/chat/text", response_model=ChatResponse)
async def chat_text(message: TextMessage):
    """Send a text message and get a complete response"""
    try:
        chatbot = await get_or_create_chatbot(message.session_id)
        
        # Connect if not already connected
        if not chatbot.is_connected:
            await chatbot.connect(modalities=["text"])
        
        # Send message and get response
        response_text = ""
        response_complete = False
        
        def text_handler(chunk: str):
            nonlocal response_text
            response_text += chunk
            print(f"Text chunk: {chunk}")  # Debug logging
        
        def complete_handler():
            nonlocal response_complete
            response_complete = True
            print("Text response completed")  # Debug logging
        
        chatbot.set_text_handler(text_handler)
        chatbot.set_response_complete_handler(complete_handler)
        
        print(f"Sending text message: {message.message}")  # Debug logging
        await chatbot.send_text_message(message.message, include_audio=False)
        await chatbot.listen_for_responses()
        
        # Wait for response completion with timeout
        timeout_counter = 0
        while not response_complete and timeout_counter < 100:  # 10 second timeout
            await asyncio.sleep(0.1)
            timeout_counter += 1
        
        print(f"Final text response: '{response_text}'")  # Debug logging
        
        # Ensure we have some response
        if not response_text.strip():
            response_text = "I'm here to help you with parenting support. How can I assist you today?"
        
        return ChatResponse(
            response=response_text,
            session_id=message.session_id
        )
        
    except Exception as e:
        print(f"Error in text chat: {str(e)}")  # Debug logging
        raise HTTPException(status_code=500, detail=str(e))

# Streaming text chat endpoint
@app.post("/chat/text/stream")
async def chat_text_stream(message: TextMessage):
    """Send a text message and get a streaming response"""
    try:
        chatbot = await get_or_create_chatbot(message.session_id)
        
        # Connect if not already connected
        if not chatbot.is_connected:
            await chatbot.connect(modalities=["text"])
        
        async def generate_response():
            response_chunks = []
            response_complete = False
            chunks_sent = 0
            
            def text_handler(chunk: str):
                if chunk.strip():  # Only add non-empty chunks
                    response_chunks.append(chunk)
                    print(f"Stream handler received chunk: '{chunk}'")  # Debug logging
            
            def complete_handler():
                nonlocal response_complete
                response_complete = True
                print(f"Stream response complete. Total chunks sent: {chunks_sent}")  # Debug logging
            
            chatbot.set_text_handler(text_handler)
            chatbot.set_response_complete_handler(complete_handler)
            
            print(f"Starting streaming response for: {message.message}")  # Debug logging
            
            # Send message
            await chatbot.send_text_message(message.message, include_audio=False)
            
            # Start listening in background
            listen_task = asyncio.create_task(chatbot.listen_for_responses())
            
            # Give the API a moment to start responding
            await asyncio.sleep(0.1)
            
            # Stream chunks as they arrive
            timeout_counter = 0
            max_timeout = 500  # 50 seconds maximum wait
            
            while not response_complete and timeout_counter < max_timeout:
                if response_chunks:
                    chunk = response_chunks.pop(0)
                    chunks_sent += 1
                    print(f"Streaming chunk {chunks_sent}: '{chunk}'")  # Debug logging
                    yield f"data: {json.dumps({'chunk': chunk, 'done': False, 'chunk_number': chunks_sent})}\n\n"
                else:
                    await asyncio.sleep(0.1)  # Wait for chunks
                    timeout_counter += 1
            
            # Wait for listen task to complete or timeout
            try:
                await asyncio.wait_for(listen_task, timeout=5.0)
            except asyncio.TimeoutError:
                print("Listen task timed out")
                listen_task.cancel()
            
            # Send any remaining chunks
            while response_chunks:
                chunk = response_chunks.pop(0)
                chunks_sent += 1
                print(f"Final chunk {chunks_sent}: '{chunk}'")  # Debug logging
                yield f"data: {json.dumps({'chunk': chunk, 'done': False, 'chunk_number': chunks_sent})}\n\n"
            
            # Send completion signal
            print(f"Streaming complete. Total chunks: {chunks_sent}")  # Debug logging
            yield f"data: {json.dumps({'chunk': '', 'done': True, 'total_chunks': chunks_sent})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Multimodal chat endpoint (text + audio)
@app.post("/chat/multimodal", response_model=ChatResponse)
async def chat_multimodal(message: TextMessage):
    """Send a text message and get both text and audio response"""
    try:
        chatbot = await get_or_create_chatbot(message.session_id)
        
        # Connect with both modalities
        if not chatbot.is_connected:
            await chatbot.connect(modalities=["text", "audio"])
        
        response_text = ""
        response_complete = False
        
        def text_handler(chunk: str):
            nonlocal response_text
            response_text += chunk
            print(f"Text chunk received: {chunk}")  # Debug logging
        
        def complete_handler():
            nonlocal response_complete
            response_complete = True
            print("Response completed")  # Debug logging
        
        chatbot.set_text_handler(text_handler)
        chatbot.set_response_complete_handler(complete_handler)
        
        print(f"Sending message: {message.message}")  # Debug logging
        await chatbot.send_text_message(message.message, include_audio=True)
        await chatbot.listen_for_responses()
        
        # Wait a bit more to ensure response is complete
        timeout_counter = 0
        while not response_complete and timeout_counter < 100:  # 10 second timeout
            await asyncio.sleep(0.1)
            timeout_counter += 1
        
        print(f"Final response text: '{response_text}'")  # Debug logging
        print(f"Audio response length: {len(chatbot.current_audio_response) if chatbot.current_audio_response else 0}")  # Debug logging
        
        # Get audio response if available
        audio_b64 = None
        if chatbot.current_audio_response and len(chatbot.current_audio_response) > 0:
            audio_b64 = base64.b64encode(chatbot.current_audio_response).decode()
            print(f"Audio encoded, length: {len(audio_b64)}")  # Debug logging
        
        # Ensure we have some response
        if not response_text.strip():
            response_text = "I'm here to help you with parenting support. How can I assist you today?"
        
        return ChatResponse(
            response=response_text,
            session_id=message.session_id,
            audio_data=audio_b64
        )
        
    except Exception as e:
        print(f"Error in multimodal chat: {str(e)}")  # Debug logging
        raise HTTPException(status_code=500, detail=str(e))

# Streaming multimodal chat endpoint (text + audio)
@app.post("/chat/multimodal/stream")
async def chat_multimodal_stream(message: TextMessage):
    """Send a text message and get streaming text and audio response"""
    try:
        chatbot = await get_or_create_chatbot(message.session_id)
        
        # Connect with both modalities
        if not chatbot.is_connected:
            await chatbot.connect(modalities=["text", "audio"])
        
        async def generate_response():
            response_chunks = []
            audio_chunks = []
            response_complete = False
            chunks_sent = 0
            
            def text_handler(chunk: str):
                if chunk.strip():  # Only add non-empty chunks
                    response_chunks.append({"type": "text", "data": chunk})
                    print(f"Multimodal stream text chunk: '{chunk}'")  # Debug logging
            
            def audio_handler(audio_chunk: bytes):
                # Convert audio chunk to base64
                audio_b64 = base64.b64encode(audio_chunk).decode()
                audio_chunks.append({"type": "audio", "data": audio_b64})
                print(f"Multimodal stream audio chunk: {len(audio_chunk)} bytes")  # Debug logging
            
            def complete_handler():
                nonlocal response_complete
                response_complete = True
                print(f"Multimodal stream complete. Total chunks sent: {chunks_sent}")  # Debug logging
            
            # Set ALL handlers before sending message
            chatbot.set_text_handler(text_handler)
            chatbot.set_audio_handler(audio_handler)
            chatbot.set_response_complete_handler(complete_handler)
            
            print(f"Starting multimodal streaming for: {message.message}")  # Debug logging
            
            # Send message with audio
            await chatbot.send_text_message(message.message, include_audio=True)
            
            # Start listening in background
            listen_task = asyncio.create_task(chatbot.listen_for_responses())
            
            # Give the API a moment to start responding
            await asyncio.sleep(0.2)  # Slightly longer delay for multimodal
            
            # Stream chunks as they arrive
            timeout_counter = 0
            max_timeout = 500  # 50 seconds maximum wait
            
            while not response_complete and timeout_counter < max_timeout:
                # Process chunks in the order they arrive
                chunks_to_send = []
                
                # Collect text chunks
                while response_chunks:
                    chunks_to_send.append(response_chunks.pop(0))
                
                # Collect audio chunks
                while audio_chunks:
                    chunks_to_send.append(audio_chunks.pop(0))
                
                # Send all collected chunks
                for chunk in chunks_to_send:
                    chunks_sent += 1
                    if chunk["type"] == "text":
                        print(f"Streaming text chunk {chunks_sent}: '{chunk['data']}'")  # Debug logging
                        yield f"data: {json.dumps({'chunk': chunk['data'], 'type': 'text', 'done': False, 'chunk_number': chunks_sent, 'session_id': message.session_id})}\n\n"
                    else:  # audio
                        print(f"Streaming audio chunk {chunks_sent}: {len(chunk['data'])} chars")  # Debug logging
                        yield f"data: {json.dumps({'chunk': chunk['data'], 'type': 'audio', 'done': False, 'chunk_number': chunks_sent, 'session_id': message.session_id})}\n\n"
                
                if not chunks_to_send:
                    await asyncio.sleep(0.1)  # Wait for chunks
                    timeout_counter += 1
            
            # Wait for listen task to complete or timeout
            try:
                await asyncio.wait_for(listen_task, timeout=5.0)
            except asyncio.TimeoutError:
                print("Multimodal listen task timed out")
                listen_task.cancel()
            
            # Send any remaining chunks
            remaining_chunks = []
            while response_chunks:
                remaining_chunks.append(response_chunks.pop(0))
            while audio_chunks:
                remaining_chunks.append(audio_chunks.pop(0))
            
            for chunk in remaining_chunks:
                chunks_sent += 1
                if chunk["type"] == "text":
                    print(f"Final text chunk {chunks_sent}: '{chunk['data']}'")  # Debug logging
                    yield f"data: {json.dumps({'chunk': chunk['data'], 'type': 'text', 'done': False, 'chunk_number': chunks_sent, 'session_id': message.session_id})}\n\n"
                else:  # audio
                    print(f"Final audio chunk {chunks_sent}")  # Debug logging
                    yield f"data: {json.dumps({'chunk': chunk['data'], 'type': 'audio', 'done': False, 'chunk_number': chunks_sent, 'session_id': message.session_id})}\n\n"
            
            # Send completion signal
            print(f"Multimodal streaming complete. Total chunks: {chunks_sent}")  # Debug logging
            yield f"data: {json.dumps({'chunk': '', 'type': 'complete', 'done': True, 'total_chunks': chunks_sent, 'session_id': message.session_id})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        print(f"Error in multimodal stream: {str(e)}")  # Debug logging
        raise HTTPException(status_code=500, detail=str(e))

# Audio input endpoint
@app.post("/chat/audio", response_model=ChatResponse)
async def chat_audio(audio_message: AudioMessage):
    """Send audio input and get response"""
    try:
        chatbot = await get_or_create_chatbot(audio_message.session_id)
        
        # Connect with audio modality
        if not chatbot.is_connected:
            await chatbot.connect(modalities=["text", "audio"])
        
        # Decode audio data
        audio_data = base64.b64decode(audio_message.audio_data)
        
        response_text = ""
        
        def text_handler(chunk: str):
            nonlocal response_text
            response_text += chunk
        
        chatbot.set_text_handler(text_handler)
        
        await chatbot.send_audio_message(audio_data)
        await chatbot.listen_for_responses()
        
        # Get audio response if available
        audio_b64 = None
        if chatbot.current_audio_response:
            audio_b64 = base64.b64encode(chatbot.current_audio_response).decode()
        
        return ChatResponse(
            response=response_text,
            session_id=audio_message.session_id,
            audio_data=audio_b64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Session management endpoints
@app.get("/session/{session_id}/stats", response_model=SessionStats)
async def get_session_stats(session_id: str):
    """Get statistics for a chat session"""
    if session_id not in chatbot_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chatbot = chatbot_sessions[session_id]
    stats = chatbot.get_conversation_stats()
    
    return SessionStats(
        session_id=session_id,
        total_messages=stats["total_messages"],
        conversation_pairs=stats["conversation_pairs"],
        user_messages=stats["user_messages"],
        assistant_messages=stats["assistant_messages"]
    )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    await cleanup_chatbot(session_id)
    return {"message": f"Session {session_id} deleted"}

@app.post("/session/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    if session_id not in chatbot_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chatbot = chatbot_sessions[session_id]
    chatbot.clear_conversation()
    return {"message": f"Session {session_id} conversation cleared"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": list(chatbot_sessions.keys()),
        "total_sessions": len(chatbot_sessions)
    }

# WebSocket endpoint for real-time streaming
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time bidirectional communication"""
    await websocket.accept()
    
    try:
        chatbot = await get_or_create_chatbot(session_id)
        
        # Connect if not already connected
        if not chatbot.is_connected:
            await chatbot.connect(modalities=["text", "audio"])
        
        def text_handler(chunk: str):
            # Send text chunks via WebSocket
            asyncio.create_task(websocket.send_json({
                "type": "text_chunk",
                "data": chunk,
                "session_id": session_id
            }))
        
        def audio_handler(audio_chunk: bytes):
            # Send audio chunks via WebSocket
            audio_b64 = base64.b64encode(audio_chunk).decode()
            asyncio.create_task(websocket.send_json({
                "type": "audio_chunk",
                "data": audio_b64,
                "session_id": session_id
            }))
        
        def complete_handler():
            # Signal response completion
            asyncio.create_task(websocket.send_json({
                "type": "response_complete",
                "session_id": session_id
            }))
        
        chatbot.set_text_handler(text_handler)
        chatbot.set_audio_handler(audio_handler)
        chatbot.set_response_complete_handler(complete_handler)
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data["type"] == "text_message":
                include_audio = data.get("include_audio", True)  # Default to multimodal for WebSocket
                await chatbot.send_text_message(data["message"], include_audio=include_audio)
                await chatbot.listen_for_responses()
            
            elif data["type"] == "audio_message":
                audio_data = base64.b64decode(data["audio_data"])
                await chatbot.send_audio_message(audio_data)
                await chatbot.listen_for_responses()
            
            elif data["type"] == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket error for session {session_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        # Clean up if needed
        pass

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all chatbot sessions on shutdown"""
    for session_id in list(chatbot_sessions.keys()):
        await cleanup_chatbot(session_id)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )