import pytest
import asyncio
import json
import base64
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestAudioProcessing:
    """Test suite for audio processing functionality"""
    
    def test_audio_format_validation(self):
        """Test audio format specifications"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Test default audio settings
            assert chatbot.audio_format == "pcm16"
            assert chatbot.sample_rate == 24000
    
    def test_audio_encoding_decoding(self):
        """Test audio base64 encoding and decoding"""
        # Create sample audio data
        sample_audio = b"RIFF\x24\x08\x00\x00WAVE"  # WAV header-like data
        
        # Test encoding
        encoded = base64.b64encode(sample_audio).decode()
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        
        # Test decoding
        decoded = base64.b64decode(encoded)
        assert decoded == sample_audio
    
    def test_audio_chunk_accumulation(self):
        """Test audio chunk accumulation during streaming"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Start with empty audio
            assert chatbot.current_audio_response == b""
            
            # Simulate receiving audio chunks
            chunk1 = b"chunk1"
            chunk2 = b"chunk2"
            chunk3 = b"chunk3"
            
            # Test manual accumulation (simulating what happens in _handle_event)
            chatbot.current_audio_response += chunk1
            chatbot.current_audio_response += chunk2
            chatbot.current_audio_response += chunk3
            
            expected = b"chunk1chunk2chunk3"
            assert chatbot.current_audio_response == expected
    
    def test_save_audio_to_file(self):
        """Test saving audio data to WAV file"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Create fake PCM audio data
            fake_audio = b"\x00\x01" * 1000  # 2000 bytes of fake 16-bit audio
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_filename = tmp_file.name
            
            try:
                # Test saving audio
                chatbot.save_audio_to_file(fake_audio, tmp_filename)
                
                # Verify file was created and has content
                assert os.path.exists(tmp_filename)
                assert os.path.getsize(tmp_filename) > 0
                
                # Basic WAV file validation
                with open(tmp_filename, 'rb') as f:
                    header = f.read(4)
                    assert header == b'RIFF'  # WAV files start with RIFF
                    
            finally:
                # Clean up
                if os.path.exists(tmp_filename):
                    os.unlink(tmp_filename)


class TestTextProcessing:
    """Test suite for text processing functionality"""
    
    def test_text_chunk_accumulation(self):
        """Test text chunk accumulation during streaming"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Start with empty text
            assert chatbot.current_text_response == ""
            
            # Simulate receiving text chunks
            chunks = ["Hello", " ", "world", "!", " How", " are", " you", "?"]
            
            for chunk in chunks:
                chatbot.current_text_response += chunk
            
            expected = "Hello world! How are you?"
            assert chatbot.current_text_response == expected
    
    def test_text_handler_callback(self):
        """Test text handler callback functionality"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            
            received_chunks = []
            
            def text_handler(chunk):
                received_chunks.append(chunk)
            
            chatbot.set_text_handler(text_handler)
            
            # Simulate text events
            chunks = ["Hello", " world"]
            for chunk in chunks:
                chatbot.text_handler(chunk)
            
            assert received_chunks == chunks
    
    def test_response_completion_callback(self):
        """Test response completion callback"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            
            completion_called = False
            
            def complete_handler():
                nonlocal completion_called
                completion_called = True
            
            chatbot.set_response_complete_handler(complete_handler)
            
            # Simulate completion
            chatbot.response_complete_handler()
            
            assert completion_called


class TestSessionManagement:
    """Test suite for session management functionality"""
    
    def test_session_isolation(self):
        """Test that sessions are properly isolated"""
        from main import chatbot_sessions
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        # Clear sessions
        chatbot_sessions.clear()
        
        with patch('main.RealtimeChatbot') as mock_chatbot_class:
            # Create different mock instances for different sessions
            mock_chatbot1 = Mock(spec=RealtimeChatbot)
            mock_chatbot2 = Mock(spec=RealtimeChatbot)
            
            mock_chatbot_class.side_effect = [mock_chatbot1, mock_chatbot2]
            
            from main import get_or_create_chatbot
            
            async def test():
                # Create two different sessions
                chatbot1 = await get_or_create_chatbot("session1")
                chatbot2 = await get_or_create_chatbot("session2")
                
                # Should be different instances
                assert chatbot1 is not chatbot2
                assert len(chatbot_sessions) == 2
                
                # Getting same session should return same instance
                chatbot1_again = await get_or_create_chatbot("session1")
                assert chatbot1 is chatbot1_again
                
                return chatbot1, chatbot2
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                chatbot1, chatbot2 = loop.run_until_complete(test())
            finally:
                loop.close()
    
    def test_session_cleanup(self):
        """Test session cleanup functionality"""
        from main import chatbot_sessions, cleanup_chatbot
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        # Setup mock chatbot
        mock_chatbot = Mock(spec=RealtimeChatbot)
        mock_chatbot.is_connected = True
        mock_chatbot.disconnect = AsyncMock()
        
        # Add to sessions
        chatbot_sessions["test_session"] = mock_chatbot
        
        async def test():
            await cleanup_chatbot("test_session")
            
            # Should have called disconnect and removed from sessions
            mock_chatbot.disconnect.assert_called_once()
            assert "test_session" not in chatbot_sessions
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test())
        finally:
            loop.close()
    
    def test_conversation_statistics(self):
        """Test conversation statistics calculation"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Add sample conversation
            conversation = [
                {"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
                {"type": "message", "role": "user", "content": [{"type": "text", "text": "How are you?"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "I'm doing well!"}]},
            ]
            
            chatbot.conversation_history = conversation
            
            stats = chatbot.get_conversation_stats()
            
            assert stats["total_messages"] == 4
            assert stats["conversation_pairs"] == 2
            assert stats["user_messages"] == 2
            assert stats["assistant_messages"] == 2


class TestErrorHandling:
    """Test suite for error handling"""
    
    def test_connection_error_handling(self):
        """Test handling of connection errors"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI') as mock_client:
            mock_client.side_effect = Exception("Network error")
            
            with pytest.raises(Exception, match="Network error"):
                chatbot = RealtimeChatbot(api_key="test_key")
    
    def test_not_connected_error_handling(self):
        """Test error handling when not connected"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            # Ensure not connected
            chatbot.is_connected = False
            
            async def test():
                with pytest.raises(RuntimeError, match="Not connected"):
                    await chatbot.send_text_message("test")
                
                with pytest.raises(RuntimeError, match="Not connected"):
                    await chatbot.send_audio_message(b"test")
                
                with pytest.raises(RuntimeError, match="Not connected"):
                    await chatbot.listen_for_responses()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(test())
            finally:
                loop.close()
    
    def test_api_error_event_handling(self):
        """Test handling of API error events"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Create error event
            error_event = Mock()
            error_event.type = "error"
            error_event.error = "API rate limit exceeded"
            
            # Test error handling (should not raise, just log)
            async def test():
                with patch('builtins.print') as mock_print:
                    await chatbot._handle_event(error_event)
                    mock_print.assert_called_with("❌ API Error: API rate limit exceeded")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(test())
            finally:
                loop.close()


class TestConfigurationValidation:
    """Test suite for configuration validation"""
    
    def test_chatbot_initialization_parameters(self):
        """Test chatbot initialization with different parameters"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            # Test default parameters
            chatbot1 = RealtimeChatbot(api_key="test_key")
            assert chatbot1.model == "gpt-4o-realtime-preview"
            assert chatbot1.voice == "alloy"
            
            # Test custom parameters
            chatbot2 = RealtimeChatbot(
                api_key="test_key",
                model="custom-model",
                voice="nova"
            )
            assert chatbot2.model == "custom-model"
            assert chatbot2.voice == "nova"
    
    def test_session_configuration(self):
        """Test session configuration parameters"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI') as mock_client:
            mock_connection = Mock()
            mock_connection.session.update = AsyncMock()
            
            mock_manager = Mock()
            mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_manager.__aexit__ = AsyncMock()
            
            mock_client.return_value.beta.realtime.connect.return_value = mock_manager
            
            chatbot = RealtimeChatbot(api_key="test_key", voice="shimmer")
            
            async def test():
                await chatbot.connect(modalities=["text", "audio"])
                
                # Verify session.update was called with correct config
                mock_connection.session.update.assert_called_once()
                call_args = mock_connection.session.update.call_args[1]
                session_config = call_args["session"]
                
                assert session_config["modalities"] == ["text", "audio"]
                assert session_config["voice"] == "shimmer"
                assert session_config["input_audio_format"] == "pcm16"
                assert session_config["output_audio_format"] == "pcm16"
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(test())
            finally:
                loop.close()


class TestDataFlow:
    """Test suite for data flow through the system"""
    
    def test_text_message_flow(self):
        """Test complete text message flow"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI') as mock_client:
            mock_connection = Mock()
            mock_connection.session.update = AsyncMock()
            mock_connection.conversation.item.create = AsyncMock()
            mock_connection.response.create = AsyncMock()
            
            mock_manager = Mock()
            mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_manager.__aexit__ = AsyncMock()
            
            mock_client.return_value.beta.realtime.connect.return_value = mock_manager
            
            chatbot = RealtimeChatbot(api_key="test_key")
            
            async def test():
                await chatbot.connect(modalities=["text"])
                await chatbot.send_text_message("Hello world", include_audio=False)
                
                # Verify conversation item was created
                mock_connection.conversation.item.create.assert_called_once()
                call_args = mock_connection.conversation.item.create.call_args[1]
                item = call_args["item"]
                
                assert item["type"] == "message"
                assert item["role"] == "user"
                assert item["content"][0]["type"] == "input_text"
                assert item["content"][0]["text"] == "Hello world"
                
                # Verify response was created
                mock_connection.response.create.assert_called_once()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(test())
            finally:
                loop.close()
    
    def test_audio_message_flow(self):
        """Test complete audio message flow"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI') as mock_client:
            mock_connection = Mock()
            mock_connection.session.update = AsyncMock()
            mock_connection.conversation.item.create = AsyncMock()
            mock_connection.response.create = AsyncMock()
            
            mock_manager = Mock()
            mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_manager.__aexit__ = AsyncMock()
            
            mock_client.return_value.beta.realtime.connect.return_value = mock_manager
            
            chatbot = RealtimeChatbot(api_key="test_key")
            
            async def test():
                await chatbot.connect(modalities=["text", "audio"])
                
                audio_data = b"fake_audio_data"
                await chatbot.send_audio_message(audio_data)
                
                # Verify conversation item was created
                mock_connection.conversation.item.create.assert_called_once()
                call_args = mock_connection.conversation.item.create.call_args[1]
                item = call_args["item"]
                
                assert item["type"] == "message"
                assert item["role"] == "user"
                assert item["content"][0]["type"] == "input_audio"
                assert "audio" in item["content"][0]
                
                # Verify response was created with multimodal config
                mock_connection.response.create.assert_called_once()
                response_call_args = mock_connection.response.create.call_args[1]
                response_config = response_call_args["response"]
                assert response_config["modalities"] == ["text", "audio"]
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(test())
            finally:
                loop.close()


if __name__ == "__main__":
    pytest.main([__file__])
