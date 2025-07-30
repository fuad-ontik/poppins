import pytest
import asyncio
import json
import base64
from unittest.mock import Mock, AsyncMock, patch
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_complete_text_conversation_flow(self):
        """Test a complete text conversation flow"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        # Mock the OpenAI client
        with patch('chatbot.multimodal_realtime.AsyncOpenAI') as mock_client:
            # Setup mock connection
            mock_connection = Mock()
            mock_connection.session.update = AsyncMock()
            mock_connection.conversation.item.create = AsyncMock()
            mock_connection.response.create = AsyncMock()
            
            # Mock events for text response
            mock_events = [
                Mock(type="session.created"),
                Mock(type="response.text.delta", delta="Hello"),
                Mock(type="response.text.delta", delta=" there!"),
                Mock(type="response.done"),
            ]
            
            async def mock_event_iterator(self):
                for event in mock_events:
                    yield event
            
            mock_connection.__aiter__ = mock_event_iterator
            
            mock_manager = Mock()
            mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_manager.__aexit__ = AsyncMock()
            
            mock_client.return_value.beta.realtime.connect.return_value = mock_manager
            
            # Test the chatbot
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Test connection
            await chatbot.connect(modalities=["text"])
            assert chatbot.is_connected
            
            # Test text handler
            received_chunks = []
            completion_called = False
            
            def text_handler(chunk):
                received_chunks.append(chunk)
                
            def complete_handler():
                nonlocal completion_called
                completion_called = True
            
            chatbot.set_text_handler(text_handler)
            chatbot.set_response_complete_handler(complete_handler)
            
            # Send message and listen
            await chatbot.send_text_message("Hello")
            await chatbot.listen_for_responses()
            
            # Verify results
            assert received_chunks == ["Hello", " there!"]
            assert completion_called
            assert chatbot.current_text_response == "Hello there!"
            
            # Test disconnect
            await chatbot.disconnect()
            assert not chatbot.is_connected
    
    @pytest.mark.asyncio
    async def test_complete_multimodal_conversation_flow(self):
        """Test a complete multimodal conversation flow"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        # Mock audio data
        mock_audio_b64 = base64.b64encode(b"fake_audio_chunk").decode()
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI') as mock_client:
            # Setup mock connection
            mock_connection = Mock()
            mock_connection.session.update = AsyncMock()
            mock_connection.conversation.item.create = AsyncMock()
            mock_connection.response.create = AsyncMock()
            
            # Mock events for multimodal response
            mock_events = [
                Mock(type="session.created"),
                Mock(type="response.audio_transcript.delta", delta="Hello"),
                Mock(type="response.audio.delta", delta=mock_audio_b64),
                Mock(type="response.audio_transcript.delta", delta=" there!"),
                Mock(type="response.audio.delta", delta=mock_audio_b64),
                Mock(type="response.done"),
            ]
            
            async def mock_event_iterator(self):
                for event in mock_events:
                    yield event
            
            mock_connection.__aiter__ = mock_event_iterator
            
            mock_manager = Mock()
            mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_manager.__aexit__ = AsyncMock()
            
            mock_client.return_value.beta.realtime.connect.return_value = mock_manager
            
            # Test the chatbot
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Test connection with audio
            await chatbot.connect(modalities=["text", "audio"])
            assert chatbot.is_connected
            
            # Test handlers
            received_text_chunks = []
            received_audio_chunks = []
            completion_called = False
            
            def text_handler(chunk):
                received_text_chunks.append(chunk)
                
            def audio_handler(chunk):
                received_audio_chunks.append(chunk)
                
            def complete_handler():
                nonlocal completion_called
                completion_called = True
            
            chatbot.set_text_handler(text_handler)
            chatbot.set_audio_handler(audio_handler)
            chatbot.set_response_complete_handler(complete_handler)
            
            # Send message with audio and listen
            await chatbot.send_text_message("Hello", include_audio=True)
            await chatbot.listen_for_responses()
            
            # Verify results
            assert received_text_chunks == ["Hello", " there!"]
            assert len(received_audio_chunks) == 2
            assert completion_called
            assert chatbot.current_text_response == "Hello there!"
            assert len(chatbot.current_audio_response) > 0
            
            # Test disconnect
            await chatbot.disconnect()
            assert not chatbot.is_connected
    
    def test_session_management_integration(self):
        """Test session management across the application"""
        from main import chatbot_sessions, get_or_create_chatbot
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        # Clear sessions first
        chatbot_sessions.clear()
        
        with patch('main.RealtimeChatbot') as mock_chatbot_class:
            # Create a factory function that returns different instances
            def create_different_mock(*args, **kwargs):
                mock = Mock(spec=RealtimeChatbot)
                mock.is_connected = False
                mock.get_conversation_stats.return_value = {
                    "total_messages": 4,
                    "conversation_pairs": 2,
                    "user_messages": 2,
                    "assistant_messages": 2
                }
                return mock
            
            mock_chatbot_class.side_effect = create_different_mock
            
            # Test session creation
            import asyncio
            
            async def test_session_creation():
                chatbot1 = await get_or_create_chatbot("session1")
                chatbot2 = await get_or_create_chatbot("session2")
                chatbot1_again = await get_or_create_chatbot("session1")
                
                # Should have 2 sessions
                assert len(chatbot_sessions) == 2
                assert chatbot1 is chatbot1_again  # Same session should return same instance
                assert chatbot1 is not chatbot2  # Different sessions should be different
                
                return chatbot1, chatbot2
            
            # Run the async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                chatbot1, chatbot2 = loop.run_until_complete(test_session_creation())
                
                # Test session statistics - each chatbot already has stats configured
                stats = chatbot1.get_conversation_stats()
                assert stats["total_messages"] == 4
                assert stats["conversation_pairs"] == 2
                
            finally:
                loop.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling across the system"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        # Test initialization error
        with patch('chatbot.multimodal_realtime.AsyncOpenAI') as mock_client:
            mock_client.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception, match="Connection failed"):
                chatbot = RealtimeChatbot(api_key="test_key")
        
        # Test operation errors with a properly initialized chatbot
        with patch('chatbot.multimodal_realtime.AsyncOpenAI') as mock_client:
            # Allow initialization to succeed
            mock_client.return_value = Mock()
            
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Test operation when not connected
            chatbot.is_connected = False
            
            with pytest.raises(RuntimeError, match="Not connected"):
                await chatbot.send_text_message("test")
                await chatbot.send_text_message("test")
                
            with pytest.raises(RuntimeError, match="Not connected"):
                await chatbot.send_audio_message(b"test")
                
            with pytest.raises(RuntimeError, match="Not connected"):
                await chatbot.listen_for_responses()
    
    def test_conversation_history_integration(self):
        """Test conversation history management"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Test empty history
            assert chatbot.get_conversation_stats()["total_messages"] == 0
            
            # Add messages manually (simulating conversation)
            for i in range(10):
                chatbot.conversation_history.append({
                    "type": "message",
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": [{"type": "text", "text": f"Message {i}"}]
                })
            
            # Test stats
            stats = chatbot.get_conversation_stats()
            assert stats["total_messages"] == 10
            assert stats["conversation_pairs"] == 5
            assert stats["user_messages"] == 5
            assert stats["assistant_messages"] == 5
            
            # Test trimming
            chatbot.trim_conversation_history(max_pairs=3)
            assert len(chatbot.conversation_history) == 6  # 3 pairs
            
            # Test clearing
            chatbot.clear_conversation()
            assert len(chatbot.conversation_history) == 0
    
    def test_audio_data_handling_integration(self):
        """Test audio data encoding/decoding integration"""
        import base64
        
        # Test data
        original_audio = b"This is fake audio data for testing purposes"
        
        # Encode to base64 (as would happen in frontend)
        encoded_audio = base64.b64encode(original_audio).decode()
        
        # Decode (as would happen in backend)
        decoded_audio = base64.b64decode(encoded_audio)
        
        # Verify round-trip
        assert original_audio == decoded_audio
        
        # Test with actual chatbot audio handling
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Simulate audio accumulation
            chatbot.current_audio_response = b""
            
            # Add chunks (as would happen during streaming)
            chunk1 = b"audio_chunk_1"
            chunk2 = b"audio_chunk_2"
            chunk3 = b"audio_chunk_3"
            
            chatbot.current_audio_response += chunk1
            chatbot.current_audio_response += chunk2
            chatbot.current_audio_response += chunk3
            
            # Verify accumulation
            expected = chunk1 + chunk2 + chunk3
            assert chatbot.current_audio_response == expected
            
            # Test base64 encoding for API response
            encoded_response = base64.b64encode(chatbot.current_audio_response).decode()
            decoded_back = base64.b64decode(encoded_response)
            assert decoded_back == expected


class TestPerformanceAndLimits:
    """Test performance characteristics and limits"""
    
    def test_conversation_history_memory_limits(self):
        """Test that conversation history doesn't grow unbounded"""
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            
            # Add many messages
            for i in range(100):
                chatbot.conversation_history.append({
                    "type": "message",
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": [{"type": "text", "text": f"Message {i}"}]
                })
            
            # Should have 100 messages
            assert len(chatbot.conversation_history) == 100
            
            # Trim to reasonable size
            chatbot.trim_conversation_history(max_pairs=5)
            
            # Should now have only 10 messages (5 pairs)
            assert len(chatbot.conversation_history) == 10
            
            # Verify the right messages were kept (most recent)
            first_kept_message = chatbot.conversation_history[0]["content"][0]["text"]
            last_kept_message = chatbot.conversation_history[-1]["content"][0]["text"]
            
            # Should keep the last 10 messages (90-99)
            assert "Message 90" in first_kept_message
            assert "Message 99" in last_kept_message
    
    def test_large_audio_data_handling(self):
        """Test handling of large audio data"""
        import base64
        
        # Create large fake audio data (1MB)
        large_audio = b"x" * (1024 * 1024)
        
        # Test base64 encoding doesn't fail
        encoded = base64.b64encode(large_audio).decode()
        decoded = base64.b64decode(encoded)
        
        assert len(decoded) == len(large_audio)
        assert decoded == large_audio
        
        # Test with chatbot
        from chatbot.multimodal_realtime import RealtimeChatbot
        
        with patch('chatbot.multimodal_realtime.AsyncOpenAI'):
            chatbot = RealtimeChatbot(api_key="test_key")
            chatbot.current_audio_response = large_audio
            
            # Should handle large audio without issues
            assert len(chatbot.current_audio_response) == 1024 * 1024


class TestMockUtilities:
    """Utilities for creating consistent mocks across tests"""
    
    @staticmethod
    def create_mock_openai_client():
        """Create a standard mock OpenAI client"""
        client = Mock()
        client.beta = Mock()
        client.beta.realtime = Mock()
        return client
    
    @staticmethod
    def create_mock_chatbot():
        """Create a standard mock chatbot"""
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
    
    @staticmethod
    def create_sample_events(event_type="text"):
        """Create sample events for testing"""
        if event_type == "text":
            return [
                Mock(type="session.created"),
                Mock(type="response.text.delta", delta="Hello"),
                Mock(type="response.text.delta", delta=" world!"),
                Mock(type="response.done"),
            ]
        elif event_type == "audio":
            mock_audio = base64.b64encode(b"fake_audio").decode()
            return [
                Mock(type="session.created"),
                Mock(type="response.audio_transcript.delta", delta="Hello"),
                Mock(type="response.audio.delta", delta=mock_audio),
                Mock(type="response.audio_transcript.delta", delta=" world!"),
                Mock(type="response.audio.delta", delta=mock_audio),
                Mock(type="response.done"),
            ]
        else:
            return []


if __name__ == "__main__":
    pytest.main([__file__])
