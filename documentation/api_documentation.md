# Poppins API Documentation

## Overview
Poppins is a realtime parent assistant API that provides multimodal chatbot capabilities using OpenAI's GPT-4o realtime model. The API supports text-only, audio-only, and multimodal (text + audio) conversations with both streaming and non-streaming responses.

**Base URL**: `http://localhost:8000`

**API Version**: 1.0.0

## Authentication
Currently, no authentication is required. The OpenAI API key is configured server-side via environment variables.

## Common Data Types

### TextMessage
```json
{
  "message": "string",
  "session_id": "string (optional, default: 'default')"
}
```

### AudioMessage
```json
{
  "audio_data": "string (base64 encoded audio)",
  "session_id": "string (optional, default: 'default')"
}
```

### ChatResponse
```json
{
  "response": "string",
  "session_id": "string",
  "audio_data": "string | null (base64 encoded audio)"
}
```

### SessionStats
```json
{
  "session_id": "string",
  "total_messages": "integer",
  "conversation_pairs": "integer",
  "user_messages": "integer",
  "assistant_messages": "integer"
}
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API service is running.

**Request**: No parameters

**Response**: 
```json
{
  "status": "healthy",
  "service": "poppins-api"
}
```

**Status Codes**:
- `200`: Service is healthy

---

### 2. Debug Connection Test

**GET** `/debug/test-connection`

Test the OpenAI connection and basic functionality.

**Request**: No parameters

**Response**:
```json
{
  "connection_successful": true,
  "response_received": true,
  "response_length": 45,
  "response_preview": "Hello! I'm here to help you with...",
  "timeout_reached": false
}
```

**Status Codes**:
- `200`: Test completed (check response for success/failure details)

---

### 3. Text Chat (Non-Streaming)

**POST** `/chat/text`

Send a text message and receive a complete text response.

**Request Body** (`TextMessage`):
```json
{
  "message": "My baby won't stop crying. What should I do?",
  "session_id": "user123"
}
```

**Response** (`ChatResponse`):
```json
{
  "response": "I understand how overwhelming it can be when your baby won't stop crying. Here are some gentle techniques you can try...",
  "session_id": "user123",
  "audio_data": null
}
```

**Status Codes**:
- `200`: Success
- `500`: Internal server error

---

### 4. Text Chat (Streaming)

**POST** `/chat/text/stream`

Send a text message and receive a streaming text response using Server-Sent Events (SSE).

**Request Body** (`TextMessage`):
```json
{
  "message": "How can I help my toddler sleep better?",
  "session_id": "user123"
}
```

**Response**: Stream of Server-Sent Events

**Event Format**:
```
data: {"chunk": "Creating", "done": false, "chunk_number": 1}

data: {"chunk": " a", "done": false, "chunk_number": 2}

data: {"chunk": " good", "done": false, "chunk_number": 3}

data: {"chunk": "", "done": true, "total_chunks": 3}
```

**Response Headers**:
- `Content-Type`: `text/plain`
- `Cache-Control`: `no-cache`
- `Connection`: `keep-alive`

**Status Codes**:
- `200`: Success (streaming)
- `500`: Internal server error

---

### 5. Multimodal Chat (Non-Streaming)

**POST** `/chat/multimodal`

Send a text message and receive both text and audio response.

**Request Body** (`TextMessage`):
```json
{
  "message": "Tell me about baby sleep schedules",
  "session_id": "user123"
}
```

**Response** (`ChatResponse`):
```json
{
  "response": "Establishing a good sleep schedule for your baby is crucial for their development...",
  "session_id": "user123",
  "audio_data": "UklGRjzqAABXQVZFZm10IBAAAAABAAEAQ..." // base64 encoded audio
}
```

**Audio Format**: PCM16, 24kHz, Mono, Base64 encoded

**Status Codes**:
- `200`: Success
- `500`: Internal server error

---

### 6. Multimodal Chat (Streaming)

**POST** `/chat/multimodal/stream`

Send a text message and receive streaming text and audio response using Server-Sent Events (SSE).

**Request Body** (`TextMessage`):
```json
{
  "message": "What are some calming techniques for babies?",
  "session_id": "user123"
}
```

**Response**: Stream of Server-Sent Events

**Text Chunk Event**:
```
data: {"chunk": "Here are", "type": "text", "done": false, "chunk_number": 1, "session_id": "user123"}
```

**Audio Chunk Event**:
```
data: {"chunk": "UklGRjzqAABXQVZFZm10...", "type": "audio", "done": false, "chunk_number": 2, "session_id": "user123"}
```

**Completion Event**:
```
data: {"chunk": "", "type": "complete", "done": true, "total_chunks": 15, "session_id": "user123"}
```

**Event Types**:
- `"text"`: Text content chunk
- `"audio"`: Audio content chunk (base64 encoded)
- `"complete"`: Response completed

**Status Codes**:
- `200`: Success (streaming)
- `500`: Internal server error

---

### 7. Audio Input Chat

**POST** `/chat/audio`

Send audio input and receive text and audio response.

**Request Body** (`AudioMessage`):
```json
{
  "audio_data": "UklGRjzqAABXQVZFZm10IBAAAAABAAEAQ...", // base64 encoded audio
  "session_id": "user123"
}
```

**Audio Input Format**: PCM16, 24kHz, Mono, Base64 encoded

**Response** (`ChatResponse`):
```json
{
  "response": "I heard you asking about feeding schedules. Here's what I recommend...",
  "session_id": "user123",
  "audio_data": "UklGRjzqAABXQVZFZm10IBAAAAABAAEAQ..." // base64 encoded audio response
}
```

**Status Codes**:
- `200`: Success
- `500`: Internal server error

---

### 8. Get Session Statistics

**GET** `/session/{session_id}/stats`

Get conversation statistics for a specific session.

**Path Parameters**:
- `session_id`: String - The session identifier

**Response** (`SessionStats`):
```json
{
  "session_id": "user123",
  "total_messages": 10,
  "conversation_pairs": 5,
  "user_messages": 5,
  "assistant_messages": 5
}
```

**Status Codes**:
- `200`: Success
- `404`: Session not found

---

### 9. Clear Session History

**POST** `/session/{session_id}/clear`

Clear the conversation history for a specific session.

**Path Parameters**:
- `session_id`: String - The session identifier

**Response**:
```json
{
  "message": "Session user123 conversation cleared"
}
```

**Status Codes**:
- `200`: Success
- `404`: Session not found

---

### 10. Delete Session

**DELETE** `/session/{session_id}`

Delete a chat session and clean up resources.

**Path Parameters**:
- `session_id`: String - The session identifier

**Response**:
```json
{
  "message": "Session user123 deleted"
}
```

**Status Codes**:
- `200`: Success

---

### 11. List Active Sessions

**GET** `/sessions`

Get a list of all active chat sessions.

**Response**:
```json
{
  "active_sessions": ["user123", "user456", "default"],
  "total_sessions": 3
}
```

**Status Codes**:
- `200`: Success

---

### 12. WebSocket Real-time Communication

**WebSocket** `/ws/{session_id}`

Establish a WebSocket connection for real-time bidirectional communication.

**Path Parameters**:
- `session_id`: String - The session identifier

**Connection URL**: `ws://localhost:8000/ws/{session_id}`

#### WebSocket Message Types

**Send Text Message**:
```json
{
  "type": "text_message",
  "message": "Hello, I need help with my baby",
  "include_audio": true // optional, defaults to true
}
```

**Send Audio Message**:
```json
{
  "type": "audio_message",
  "audio_data": "UklGRjzqAABXQVZFZm10..." // base64 encoded audio
}
```

**Ping Message**:
```json
{
  "type": "ping"
}
```

#### WebSocket Response Types

**Text Chunk**:
```json
{
  "type": "text_chunk",
  "data": "Here's some advice...",
  "session_id": "user123"
}
```

**Audio Chunk**:
```json
{
  "type": "audio_chunk",
  "data": "UklGRjzqAABXQVZFZm10...", // base64 encoded audio
  "session_id": "user123"
}
```

**Response Complete**:
```json
{
  "type": "response_complete",
  "session_id": "user123"
}
```

**Pong Response**:
```json
{
  "type": "pong"
}
```

**Error**:
```json
{
  "type": "error",
  "message": "Error description"
}
```

---

## Error Handling

### Common Error Response Format
```json
{
  "detail": "Error description"
}
```

### Status Codes
- `200`: Success
- `404`: Resource not found
- `500`: Internal server error

## Audio Format Specifications

### Input Audio Format
- **Encoding**: PCM16
- **Sample Rate**: 24,000 Hz
- **Channels**: Mono (1 channel)
- **Encoding**: Base64

### Output Audio Format
- **Encoding**: PCM16
- **Sample Rate**: 24,000 Hz
- **Channels**: Mono (1 channel)
- **Encoding**: Base64

## Rate Limiting
Currently, no rate limiting is implemented. Consider implementing rate limiting for production use.

## Session Management
- Sessions are created automatically when first accessed
- Sessions persist in memory until server restart or explicit deletion
- Each session maintains independent conversation history
- Session IDs are strings and should be unique per user/conversation

## Best Practices

### Frontend Integration

1. **Session Management**: Use unique session IDs for different users or conversation threads
2. **Streaming**: Use EventSource API for Server-Sent Events endpoints
3. **Audio Handling**: Decode base64 audio data and play using Web Audio API or HTML5 audio elements
4. **Error Handling**: Always handle potential 500 errors and connection failures
5. **WebSocket**: Implement reconnection logic for WebSocket connections

### Example Frontend Code

#### Using Streaming Text Endpoint
```javascript
const eventSource = new EventSource('/chat/text/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: "Hello",
    session_id: "user123"
  })
});

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.done) {
    eventSource.close();
  } else {
    console.log('Chunk:', data.chunk);
  }
};
```

#### Using WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/user123');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'text_chunk':
      console.log('Text:', data.data);
      break;
    case 'audio_chunk':
      // Decode and play audio
      playAudio(data.data);
      break;
    case 'response_complete':
      console.log('Response complete');
      break;
  }
};

// Send message
ws.send(JSON.stringify({
  type: 'text_message',
  message: 'Hello',
  include_audio: true
}));
```

#### Audio Playback
```javascript
function playAudio(base64Audio) {
  const audioData = atob(base64Audio);
  const audioArray = new Uint8Array(audioData.length);
  for (let i = 0; i < audioData.length; i++) {
    audioArray[i] = audioData.charCodeAt(i);
  }
  
  const audioContext = new AudioContext();
  audioContext.decodeAudioData(audioArray.buffer)
    .then(audioBuffer => {
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      source.start();
    });
}
```

## Development Notes

- Server runs on `http://localhost:8000` by default
- CORS is enabled for all origins (configure for production)
- Debug logging is enabled for development
- Hot reload is enabled when running with `uvicorn --reload`

## Production Considerations

1. **Security**: Implement proper authentication and authorization
2. **Rate Limiting**: Add rate limiting to prevent abuse
3. **CORS**: Configure CORS for specific domains only
4. **Logging**: Implement proper logging and monitoring
5. **Error Handling**: Add more detailed error responses
6. **Audio Validation**: Validate audio format and size limits
7. **Session Persistence**: Consider using Redis or database for session storage
8. **Load Balancing**: Sessions are stored in memory, consider sticky sessions or external storage
