import pytest
import asyncio
import json
import base64
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chatbot.multimodal_realtime import RealtimeChatbot


class MockOpenAIEvent:
    """Mock OpenAI event for testing"""
    def __init__(self, event_type: str, delta: str = "", transcript: str = "", error: str = ""):
        self.type = event_type
        self.delta = delta
        self.transcript = transcript
        self.error = error


class MockOpenAIConnection:
    """Mock OpenAI connection for testing"""
    def __init__(self, events: list = None):
        self.events = events or []
        self.session_config = None
        self.items_created = []
        self.responses_created = []
        
        # Create persistent mocks
        self._session_mock = Mock()
        self._session_mock.update = AsyncMock()  # Use AsyncMock since it's awaited
        
        self._conversation_mock = Mock()
        item_mock = Mock()
        item_mock.create = AsyncMock(side_effect=self._store_item)
        self._conversation_mock.item = item_mock
        
        self._response_mock = Mock()
        self._response_mock.create = AsyncMock(side_effect=self._store_response)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def __aiter__(self):
        for event in self.events:
            yield event
            
    @property
    def session(self):
        return self._session_mock
        
    @property
    def conversation(self):
        return self._conversation_mock
        
    @property
    def response(self):
        return self._response_mock
        
    async def _store_item(self, item):
        self.items_created.append(item)
        
    async def _store_response(self, response=None):
        self.responses_created.append(response)


@pytest.fixture
def mock_openai_client():
    """Fixture providing a mock OpenAI client"""
    client = Mock()
    client.beta = Mock()
    client.beta.realtime = Mock()
    return client


@pytest.fixture
def sample_text_events():
    """Fixture providing sample text response events"""
    return [
        MockOpenAIEvent("session.created"),
        MockOpenAIEvent("session.updated"),
        MockOpenAIEvent("conversation.item.created"),
        MockOpenAIEvent("response.created"),
        MockOpenAIEvent("response.output_item.added"),
        MockOpenAIEvent("conversation.item.created"),
        MockOpenAIEvent("response.content_part.added"),
        MockOpenAIEvent("response.text.delta", delta="Hello"),
        MockOpenAIEvent("response.text.delta", delta=" there!"),
        MockOpenAIEvent("response.text.delta", delta=" How"),
        MockOpenAIEvent("response.text.delta", delta=" can"),
        MockOpenAIEvent("response.text.delta", delta=" I"),
        MockOpenAIEvent("response.text.delta", delta=" help?"),
        MockOpenAIEvent("response.text.done"),
        MockOpenAIEvent("response.content_part.done"),
        MockOpenAIEvent("response.output_item.done"),
        MockOpenAIEvent("response.done"),
    ]


@pytest.fixture
def sample_audio_events():
    """Fixture providing sample audio response events"""
    # Create some mock base64 audio data
    mock_audio = base64.b64encode(b"fake_audio_data_chunk").decode()
    
    return [
        MockOpenAIEvent("session.created"),
        MockOpenAIEvent("session.updated"),
        MockOpenAIEvent("conversation.item.created"),
        MockOpenAIEvent("response.created"),
        MockOpenAIEvent("response.output_item.added"),
        MockOpenAIEvent("conversation.item.created"),
        MockOpenAIEvent("response.content_part.added"),
        MockOpenAIEvent("response.audio_transcript.delta", delta="Hello"),
        MockOpenAIEvent("response.audio_transcript.delta", delta=" there!"),
        MockOpenAIEvent("response.audio.delta", delta=mock_audio),
        MockOpenAIEvent("response.audio_transcript.delta", delta=" How"),
        MockOpenAIEvent("response.audio.delta", delta=mock_audio),
        MockOpenAIEvent("response.audio_transcript.delta", delta=" can"),
        MockOpenAIEvent("response.audio.delta", delta=mock_audio),
        MockOpenAIEvent("response.audio_transcript.delta", delta=" I"),
        MockOpenAIEvent("response.audio.delta", delta=mock_audio),
        MockOpenAIEvent("response.audio_transcript.delta", delta=" help?"),
        MockOpenAIEvent("response.audio.delta", delta=mock_audio),
        MockOpenAIEvent("response.audio.done"),
        MockOpenAIEvent("response.audio_transcript.done"),
        MockOpenAIEvent("response.content_part.done"),
        MockOpenAIEvent("response.output_item.done"),
        MockOpenAIEvent("response.done"),
    ]


@pytest.fixture
def chatbot_instance(mock_openai_client):
    """Fixture providing a RealtimeChatbot instance with mocked client"""
    with patch('chatbot.multimodal_realtime.AsyncOpenAI') as mock_client_class:
        mock_client_class.return_value = mock_openai_client
        chatbot = RealtimeChatbot(
            api_key="test_api_key",
            model="gpt-4o-realtime-preview",
            voice="alloy"
        )
        chatbot.client = mock_openai_client
        return chatbot


@pytest.fixture
def mock_connection(chatbot_instance):
    """Fixture providing a mock connection"""
    connection = MockOpenAIConnection()
    chatbot_instance.connection = connection
    chatbot_instance.connection_manager = Mock()
    chatbot_instance.connection_manager.__aenter__ = AsyncMock(return_value=connection)
    chatbot_instance.connection_manager.__aexit__ = AsyncMock()
    return connection


class TestRealtimeChatbot:
    """Test suite for RealtimeChatbot class"""
    
    def test_initialization(self, chatbot_instance):
        """Test chatbot initialization"""
        assert chatbot_instance.model == "gpt-4o-realtime-preview"
        assert chatbot_instance.voice == "alloy"
        assert chatbot_instance.audio_format == "pcm16"
        assert chatbot_instance.sample_rate == 24000
        assert not chatbot_instance.is_connected
        assert chatbot_instance.conversation_history == []
        assert chatbot_instance.current_text_response == ""
        assert chatbot_instance.current_audio_response == b""
    
    def test_handler_setters(self, chatbot_instance):
        """Test handler setter methods"""
        text_handler = Mock()
        audio_handler = Mock()
        complete_handler = Mock()
        
        chatbot_instance.set_text_handler(text_handler)
        chatbot_instance.set_audio_handler(audio_handler)
        chatbot_instance.set_response_complete_handler(complete_handler)
        
        assert chatbot_instance.text_handler == text_handler
        assert chatbot_instance.audio_handler == audio_handler
        assert chatbot_instance.response_complete_handler == complete_handler
    
    @pytest.mark.asyncio
    async def test_connect_text_only(self, chatbot_instance, mock_openai_client):
        """Test connecting with text-only modality"""
        mock_connection = MockOpenAIConnection()
        mock_openai_client.beta.realtime.connect.return_value = Mock()
        mock_openai_client.beta.realtime.connect.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        
        chatbot_instance.connection_manager = mock_openai_client.beta.realtime.connect.return_value
        
        await chatbot_instance.connect(modalities=["text"])
        
        assert chatbot_instance.is_connected
        assert chatbot_instance.connection == mock_connection
        mock_connection.session.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_multimodal(self, chatbot_instance, mock_openai_client):
        """Test connecting with both text and audio modalities"""
        mock_connection = MockOpenAIConnection()
        mock_openai_client.beta.realtime.connect.return_value = Mock()
        mock_openai_client.beta.realtime.connect.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        
        chatbot_instance.connection_manager = mock_openai_client.beta.realtime.connect.return_value
        
        await chatbot_instance.connect(modalities=["text", "audio"])
        
        assert chatbot_instance.is_connected
        assert chatbot_instance.connection == mock_connection
        mock_connection.session.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, chatbot_instance):
        """Test disconnecting from the API"""
        mock_manager = AsyncMock()
        chatbot_instance.connection_manager = mock_manager
        chatbot_instance.is_connected = True
        
        await chatbot_instance.disconnect()
        
        assert not chatbot_instance.is_connected
        assert chatbot_instance.connection is None
        mock_manager.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_text_message(self, chatbot_instance, mock_connection):
        """Test sending a text message"""
        chatbot_instance.is_connected = True
        
        await chatbot_instance.send_text_message("Hello, test message", include_audio=False)
        
        assert len(mock_connection.items_created) == 1
        assert len(mock_connection.responses_created) == 1
        
        item = mock_connection.items_created[0]
        assert item["type"] == "message"
        assert item["role"] == "user"
        assert item["content"][0]["type"] == "input_text"
        assert item["content"][0]["text"] == "Hello, test message"
    
    @pytest.mark.asyncio
    async def test_send_text_message_with_audio(self, chatbot_instance, mock_connection):
        """Test sending a text message requesting audio response"""
        chatbot_instance.is_connected = True
        
        await chatbot_instance.send_text_message("Hello, test message", include_audio=True)
        
        assert len(mock_connection.items_created) == 1
        assert len(mock_connection.responses_created) == 1
        
        response_config = mock_connection.responses_created[0]
        assert response_config is not None
        assert "modalities" in response_config
        assert response_config["modalities"] == ["text", "audio"]
    
    @pytest.mark.asyncio
    async def test_send_audio_message(self, chatbot_instance, mock_connection):
        """Test sending an audio message"""
        chatbot_instance.is_connected = True
        audio_data = b"fake_audio_data"
        
        await chatbot_instance.send_audio_message(audio_data)
        
        assert len(mock_connection.items_created) == 1
        assert len(mock_connection.responses_created) == 1
        
        item = mock_connection.items_created[0]
        assert item["type"] == "message"
        assert item["role"] == "user"
        assert item["content"][0]["type"] == "input_audio"
        # Check that audio is base64 encoded
        assert "audio" in item["content"][0]
    
    @pytest.mark.asyncio
    async def test_listen_for_text_responses(self, chatbot_instance, sample_text_events):
        """Test listening for text-only responses"""
        mock_connection = MockOpenAIConnection(events=sample_text_events)
        chatbot_instance.connection = mock_connection
        chatbot_instance.is_connected = True
        
        text_chunks = []
        completion_called = False
        
        def text_handler(chunk):
            text_chunks.append(chunk)
            
        def complete_handler():
            nonlocal completion_called
            completion_called = True
        
        chatbot_instance.set_text_handler(text_handler)
        chatbot_instance.set_response_complete_handler(complete_handler)
        
        await chatbot_instance.listen_for_responses()
        
        assert text_chunks == ["Hello", " there!", " How", " can", " I", " help?"]
        assert completion_called
        assert chatbot_instance.current_text_response == "Hello there! How can I help?"
    
    @pytest.mark.asyncio
    async def test_listen_for_audio_responses(self, chatbot_instance, sample_audio_events):
        """Test listening for audio responses with transcript"""
        mock_connection = MockOpenAIConnection(events=sample_audio_events)
        chatbot_instance.connection = mock_connection
        chatbot_instance.is_connected = True
        
        text_chunks = []
        audio_chunks = []
        completion_called = False
        
        def text_handler(chunk):
            text_chunks.append(chunk)
            
        def audio_handler(chunk):
            audio_chunks.append(chunk)
            
        def complete_handler():
            nonlocal completion_called
            completion_called = True
        
        chatbot_instance.set_text_handler(text_handler)
        chatbot_instance.set_audio_handler(audio_handler)
        chatbot_instance.set_response_complete_handler(complete_handler)
        
        await chatbot_instance.listen_for_responses()
        
        assert text_chunks == ["Hello", " there!", " How", " can", " I", " help?"]
        assert len(audio_chunks) == 5  # 5 audio chunks in sample events
        assert completion_called
        assert chatbot_instance.current_text_response == "Hello there! How can I help?"
        assert len(chatbot_instance.current_audio_response) > 0
    
    @pytest.mark.asyncio
    async def test_handle_error_event(self, chatbot_instance):
        """Test handling error events"""
        error_event = MockOpenAIEvent("error", error="Test error message")
        
        with patch('builtins.print') as mock_print:
            await chatbot_instance._handle_event(error_event)
            mock_print.assert_called_with("❌ API Error: Test error message")
    
    def test_conversation_history_management(self, chatbot_instance):
        """Test conversation history management"""
        # Add some messages to history
        for i in range(15):
            chatbot_instance.conversation_history.append({
                "type": "message",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": [{"type": "text", "text": f"Message {i}"}]
            })
        
        # Test trimming
        chatbot_instance.trim_conversation_history(max_pairs=3)
        
        # Should have 6 messages (3 pairs)
        assert len(chatbot_instance.conversation_history) == 6
    
    def test_get_conversation_stats(self, chatbot_instance):
        """Test getting conversation statistics"""
        # Add some messages
        chatbot_instance.conversation_history = [
            {"type": "message", "role": "user", "content": []},
            {"type": "message", "role": "assistant", "content": []},
            {"type": "message", "role": "user", "content": []},
            {"type": "message", "role": "assistant", "content": []},
        ]
        
        stats = chatbot_instance.get_conversation_stats()
        
        assert stats["total_messages"] == 4
        assert stats["conversation_pairs"] == 2
        assert stats["user_messages"] == 2
        assert stats["assistant_messages"] == 2
    
    def test_clear_conversation(self, chatbot_instance):
        """Test clearing conversation history"""
        chatbot_instance.conversation_history = [{"test": "message"}]
        
        chatbot_instance.clear_conversation()
        
        assert chatbot_instance.conversation_history == []
    
    @pytest.mark.asyncio
    async def test_send_message_not_connected(self, chatbot_instance):
        """Test sending message when not connected raises error"""
        chatbot_instance.is_connected = False
        
        with pytest.raises(RuntimeError, match="Not connected"):
            await chatbot_instance.send_text_message("test")
        
        with pytest.raises(RuntimeError, match="Not connected"):
            await chatbot_instance.send_audio_message(b"test")
    
    @pytest.mark.asyncio
    async def test_listen_not_connected(self, chatbot_instance):
        """Test listening when not connected raises error"""
        chatbot_instance.is_connected = False
        
        with pytest.raises(RuntimeError, match="Not connected"):
            await chatbot_instance.listen_for_responses()


if __name__ == "__main__":
    pytest.main([__file__])
