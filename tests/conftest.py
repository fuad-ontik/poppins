# Pytest configuration file for Poppins testing
import os
import sys
import pytest
from unittest.mock import Mock, AsyncMock

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables and configurations"""
    os.environ["TESTING"] = "1"
    os.environ["OPENAI_API_KEY"] = "test_api_key_for_testing"
    os.environ["OPENAI_MODEL"] = "gpt-4o-realtime-preview"
    os.environ["OPENAI_VOICE"] = "alloy"


@pytest.fixture
def mock_openai_client():
    """Fixture providing a mock OpenAI client"""
    client = Mock()
    client.beta = Mock()
    client.beta.realtime = Mock()
    return client


@pytest.fixture
def mock_chatbot():
    """Fixture providing a mock chatbot instance"""
    from chatbot.multimodal_realtime import RealtimeChatbot
    
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
        "total_messages": 0,
        "conversation_pairs": 0,
        "user_messages": 0,
        "assistant_messages": 0
    })
    
    return chatbot


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear chatbot sessions before and after each test"""
    from main import chatbot_sessions
    chatbot_sessions.clear()
    yield
    chatbot_sessions.clear()


@pytest.fixture
def sample_audio_data():
    """Fixture providing sample audio data"""
    import base64
    # Create fake audio data
    fake_audio = b"RIFF\x24\x08\x00\x00WAVE" + b"\x00" * 1000
    return base64.b64encode(fake_audio).decode()


@pytest.fixture
def sample_text_message():
    """Fixture providing sample text message"""
    return {
        "message": "Hello, I need help with my baby's sleep schedule",
        "session_id": "test_session"
    }


@pytest.fixture
def sample_audio_message(sample_audio_data):
    """Fixture providing sample audio message"""
    return {
        "audio_data": sample_audio_data,
        "session_id": "test_session"
    }


@pytest.fixture
def client():
    """Fixture providing FastAPI test client"""
    from fastapi.testclient import TestClient
    from main import app
    
    # Override environment for testing
    import os
    os.environ["TESTING"] = "1"
    os.environ["OPENAI_API_KEY"] = "test_api_key_for_testing"
    
    return TestClient(app)
