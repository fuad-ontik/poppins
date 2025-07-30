import pytest
import asyncio
import json
import base64
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional
import os
import sys
from fastapi.testclient import TestClient
from fastapi import status

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app, chatbot_sessions, get_or_create_chatbot
from chatbot.multimodal_realtime import RealtimeChatbot


class TestFastAPIEndpoints:
    """Test suite for FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_chatbot(self):
        """Mock chatbot fixture"""
        chatbot = Mock(spec=RealtimeChatbot)
        chatbot.is_connected = False
        chatbot.current_text_response = ""
        chatbot.current_audio_response = b""
        chatbot.conversation_history = []
        
        # Mock async methods
        chatbot.connect = AsyncMock()
        chatbot.send_text_message = AsyncMock()
        chatbot.send_audio_message = AsyncMock()
        chatbot.listen_for_responses = AsyncMock()
        chatbot.disconnect = AsyncMock()
        chatbot.clear_conversation = Mock()
        chatbot.get_conversation_stats = Mock(return_value={
            "total_messages": 4,
            "conversation_pairs": 2,
            "user_messages": 2,
            "assistant_messages": 2
        })
        
        return chatbot
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test"""
        # Clear sessions before each test
        chatbot_sessions.clear()
        yield
        # Clear sessions after each test
        chatbot_sessions.clear()
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "healthy", "service": "poppins-api"}
    
    @patch('main.RealtimeChatbot')
    def test_debug_test_connection_success(self, mock_chatbot_class, client):
        """Test debug connection endpoint - success case"""
        mock_chatbot = Mock()
        mock_chatbot.connect = AsyncMock()
        mock_chatbot.send_text_message = AsyncMock()
        mock_chatbot.listen_for_responses = AsyncMock()
        mock_chatbot.disconnect = AsyncMock()
        mock_chatbot.set_text_handler = Mock()
        mock_chatbot.set_response_complete_handler = Mock()
        
        mock_chatbot_class.return_value = mock_chatbot
        
        # Mock the text handler behavior
        def mock_listen():
            # Simulate receiving text
            handler = mock_chatbot.set_text_handler.call_args[0][0]
            complete_handler = mock_chatbot.set_response_complete_handler.call_args[0][0]
            handler("Test response")
            complete_handler()
        
        mock_chatbot.listen_for_responses.side_effect = mock_listen
        
        response = client.get("/debug/test-connection")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["connection_successful"] is True
        assert "response_length" in data
        assert "response_preview" in data
    
    @patch('main.RealtimeChatbot')
    def test_debug_test_connection_failure(self, mock_chatbot_class, client):
        """Test debug connection endpoint - failure case"""
        mock_chatbot_class.side_effect = Exception("Connection failed")
        
        response = client.get("/debug/test-connection")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["connection_successful"] is False
        assert "error" in data
        assert "error_type" in data
    
    @patch('main.get_or_create_chatbot')
    def test_chat_text_success(self, mock_get_chatbot, client, mock_chatbot):
        """Test text chat endpoint - success case"""
        mock_get_chatbot.return_value = mock_chatbot
        
        # Mock text response
        def mock_listen():
            handler = mock_chatbot.set_text_handler.call_args[0][0]
            complete_handler = mock_chatbot.set_response_complete_handler.call_args[0][0]
            handler("Hello! I'm here to help with parenting support.")
            complete_handler()
        
        mock_chatbot.listen_for_responses.side_effect = mock_listen
        
        response = client.post("/chat/text", json={
            "message": "Hello, I need help with my baby",
            "session_id": "test_session"
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data
        assert data["session_id"] == "test_session"
        assert data["audio_data"] is None
    
    @patch('main.get_or_create_chatbot')
    def test_chat_text_stream(self, mock_get_chatbot, client, mock_chatbot):
        """Test text chat streaming endpoint"""
        mock_get_chatbot.return_value = mock_chatbot
        
        # Mock streaming response
        def mock_listen():
            handler = mock_chatbot.set_text_handler.call_args[0][0]
            complete_handler = mock_chatbot.set_response_complete_handler.call_args[0][0]
            handler("Hello")
            handler(" there!")
            complete_handler()
        
        mock_chatbot.listen_for_responses.side_effect = mock_listen
        
        response = client.post("/chat/text/stream", json={
            "message": "Hello",
            "session_id": "test_session"
        })
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        # Check that we get streaming data
        content = response.text
        assert "data:" in content
        assert "chunk" in content
    
    @patch('main.get_or_create_chatbot')
    def test_chat_multimodal_success(self, mock_get_chatbot, client, mock_chatbot):
        """Test multimodal chat endpoint - success case"""
        mock_get_chatbot.return_value = mock_chatbot
        
        # Mock multimodal response with audio
        mock_audio_data = base64.b64encode(b"fake_audio_data").decode()
        mock_chatbot.current_audio_response = base64.b64decode(mock_audio_data)
        
        def mock_listen():
            handler = mock_chatbot.set_text_handler.call_args[0][0]
            complete_handler = mock_chatbot.set_response_complete_handler.call_args[0][0]
            handler("Here's some parenting advice with audio!")
            complete_handler()
        
        mock_chatbot.listen_for_responses.side_effect = mock_listen
        
        response = client.post("/chat/multimodal", json={
            "message": "Tell me about baby sleep",
            "session_id": "test_session"
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data
        assert data["session_id"] == "test_session"
        assert data["audio_data"] is not None
    
    @patch('main.get_or_create_chatbot')
    def test_chat_multimodal_stream(self, mock_get_chatbot, client, mock_chatbot):
        """Test multimodal chat streaming endpoint"""
        mock_get_chatbot.return_value = mock_chatbot
        
        # Mock multimodal streaming response
        def mock_listen():
            text_handler = mock_chatbot.set_text_handler.call_args[0][0]
            audio_handler = mock_chatbot.set_audio_handler.call_args[0][0]
            complete_handler = mock_chatbot.set_response_complete_handler.call_args[0][0]
            
            text_handler("Hello")
            audio_handler(b"fake_audio_chunk")
            text_handler(" there!")
            complete_handler()
        
        mock_chatbot.listen_for_responses.side_effect = mock_listen
        
        response = client.post("/chat/multimodal/stream", json={
            "message": "Tell me about feeding schedules",
            "session_id": "test_session"
        })
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        # Check for both text and audio chunks in streaming response
        content = response.text
        assert "data:" in content
        assert '"type": "text"' in content or '"type": "audio"' in content
    
    @patch('main.get_or_create_chatbot')
    def test_chat_audio_success(self, mock_get_chatbot, client, mock_chatbot):
        """Test audio chat endpoint - success case"""
        mock_get_chatbot.return_value = mock_chatbot
        
        # Mock audio response
        mock_audio_data = base64.b64encode(b"fake_audio_response").decode()
        mock_chatbot.current_audio_response = base64.b64decode(mock_audio_data)
        
        def mock_listen():
            handler = mock_chatbot.set_text_handler.call_args[0][0]
            handler("I understood your audio message about sleep schedules.")
        
        mock_chatbot.listen_for_responses.side_effect = mock_listen
        
        # Create fake audio input
        audio_input = base64.b64encode(b"fake_audio_input").decode()
        
        response = client.post("/chat/audio", json={
            "audio_data": audio_input,
            "session_id": "test_session"
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data
        assert data["session_id"] == "test_session"
        assert data["audio_data"] is not None
    
    def test_get_session_stats_not_found(self, client):
        """Test get session stats - session not found"""
        response = client.get("/session/nonexistent/stats")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Session not found" in response.json()["detail"]
    
    @patch('main.get_or_create_chatbot')
    def test_get_session_stats_success(self, mock_get_chatbot, client, mock_chatbot):
        """Test get session stats - success case"""
        # Add chatbot to sessions
        chatbot_sessions["test_session"] = mock_chatbot
        
        response = client.get("/session/test_session/stats")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] == "test_session"
        assert "total_messages" in data
        assert "conversation_pairs" in data
        assert "user_messages" in data
        assert "assistant_messages" in data
    
    def test_clear_session_not_found(self, client):
        """Test clear session - session not found"""
        response = client.post("/session/nonexistent/clear")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @patch('main.get_or_create_chatbot')
    def test_clear_session_success(self, mock_get_chatbot, client, mock_chatbot):
        """Test clear session - success case"""
        chatbot_sessions["test_session"] = mock_chatbot
        
        response = client.post("/session/test_session/clear")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "conversation cleared" in data["message"]
        mock_chatbot.clear_conversation.assert_called_once()
    
    @patch('main.cleanup_chatbot')
    def test_delete_session(self, mock_cleanup, client):
        """Test delete session endpoint"""
        mock_cleanup.return_value = None
        
        response = client.delete("/session/test_session")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "deleted" in data["message"]
    
    def test_list_sessions_empty(self, client):
        """Test list sessions - empty"""
        response = client.get("/sessions")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["active_sessions"] == []
        assert data["total_sessions"] == 0
    
    def test_list_sessions_with_data(self, client, mock_chatbot):
        """Test list sessions - with data"""
        chatbot_sessions["session1"] = mock_chatbot
        chatbot_sessions["session2"] = mock_chatbot
        
        response = client.get("/sessions")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["active_sessions"]) == 2
        assert data["total_sessions"] == 2
        assert "session1" in data["active_sessions"]
        assert "session2" in data["active_sessions"]
    
    def test_invalid_json_request(self, client):
        """Test invalid JSON request"""
        response = client.post("/chat/text", 
                              data="invalid json",
                              headers={"content-type": "application/json"})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_required_fields(self, client):
        """Test missing required fields"""
        response = client.post("/chat/text", json={
            "session_id": "test"
            # missing 'message' field
        })
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch('main.get_or_create_chatbot')
    def test_chat_endpoint_exception_handling(self, mock_get_chatbot, client):
        """Test exception handling in chat endpoints"""
        mock_get_chatbot.side_effect = Exception("Database error")
        
        response = client.post("/chat/text", json={
            "message": "Hello",
            "session_id": "test"
        })
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestWebSocketEndpoint:
    """Test suite for WebSocket endpoint"""
    
    @pytest.fixture
    def mock_chatbot(self):
        """Mock chatbot for WebSocket tests"""
        chatbot = Mock(spec=RealtimeChatbot)
        chatbot.is_connected = False
        chatbot.connect = AsyncMock()
        chatbot.send_text_message = AsyncMock()
        chatbot.send_audio_message = AsyncMock()
        chatbot.listen_for_responses = AsyncMock()
        chatbot.set_text_handler = Mock()
        chatbot.set_audio_handler = Mock()
        chatbot.set_response_complete_handler = Mock()
        return chatbot
    
    @patch('main.get_or_create_chatbot')
    def test_websocket_connection(self, mock_get_chatbot, mock_chatbot):
        """Test WebSocket connection establishment"""
        mock_get_chatbot.return_value = mock_chatbot
        
        with TestClient(app) as client:
            with client.websocket_connect("/ws/test_session") as websocket:
                # Connection should be established
                assert websocket is not None
                mock_chatbot.connect.assert_called_once_with(modalities=["text", "audio"])
    
    @patch('main.get_or_create_chatbot')
    def test_websocket_text_message(self, mock_get_chatbot, mock_chatbot):
        """Test sending text message via WebSocket"""
        mock_get_chatbot.return_value = mock_chatbot
        
        with TestClient(app) as client:
            with client.websocket_connect("/ws/test_session") as websocket:
                # Send text message
                websocket.send_json({
                    "type": "text_message",
                    "message": "Hello WebSocket",
                    "include_audio": False
                })
                
                mock_chatbot.send_text_message.assert_called_once_with("Hello WebSocket", include_audio=False)
                mock_chatbot.listen_for_responses.assert_called_once()
    
    @patch('main.get_or_create_chatbot')
    def test_websocket_audio_message(self, mock_get_chatbot, mock_chatbot):
        """Test sending audio message via WebSocket"""
        mock_get_chatbot.return_value = mock_chatbot
        
        audio_data = base64.b64encode(b"fake_audio").decode()
        
        with TestClient(app) as client:
            with client.websocket_connect("/ws/test_session") as websocket:
                # Send audio message
                websocket.send_json({
                    "type": "audio_message",
                    "audio_data": audio_data
                })
                
                mock_chatbot.send_audio_message.assert_called_once()
                mock_chatbot.listen_for_responses.assert_called_once()
    
    @patch('main.get_or_create_chatbot')
    def test_websocket_ping_pong(self, mock_get_chatbot, mock_chatbot):
        """Test ping/pong functionality"""
        mock_get_chatbot.return_value = mock_chatbot
        
        with TestClient(app) as client:
            with client.websocket_connect("/ws/test_session") as websocket:
                # Send ping
                websocket.send_json({"type": "ping"})
                
                # Receive pong
                response = websocket.receive_json()
                assert response["type"] == "pong"


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    @patch('main.RealtimeChatbot')
    @pytest.mark.asyncio
    async def test_get_or_create_chatbot_new(self, mock_chatbot_class):
        """Test creating new chatbot session"""
        mock_chatbot = Mock()
        mock_chatbot_class.return_value = mock_chatbot
        
        chatbot = await get_or_create_chatbot("new_session")
        
        assert chatbot == mock_chatbot
        assert "new_session" in chatbot_sessions
        assert chatbot_sessions["new_session"] == mock_chatbot
    
    @pytest.mark.asyncio
    async def test_get_or_create_chatbot_existing(self, mock_chatbot):
        """Test retrieving existing chatbot session"""
        chatbot_sessions["existing_session"] = mock_chatbot
        
        chatbot = await get_or_create_chatbot("existing_session")
        
        assert chatbot == mock_chatbot


class TestDataValidation:
    """Test suite for data validation"""
    
    def test_text_message_validation(self, client):
        """Test TextMessage validation"""
        # Valid message
        response = client.post("/chat/text", json={
            "message": "Valid message",
            "session_id": "test"
        })
        assert response.status_code != status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Invalid message (empty string should be allowed, but null should not)
        response = client.post("/chat/text", json={
            "message": "",
            "session_id": "test"
        })
        assert response.status_code != status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Missing message field
        response = client.post("/chat/text", json={
            "session_id": "test"
        })
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_audio_message_validation(self, client):
        """Test AudioMessage validation"""
        valid_audio = base64.b64encode(b"fake_audio").decode()
        
        # Valid audio message
        response = client.post("/chat/audio", json={
            "audio_data": valid_audio,
            "session_id": "test"
        })
        # This might fail due to actual processing, but validation should pass
        assert response.status_code != status.HTTP_422_UNPROCESSABLE_ENTITY or response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        # Missing audio_data field
        response = client.post("/chat/audio", json={
            "session_id": "test"
        })
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


if __name__ == "__main__":
    pytest.main([__file__])
