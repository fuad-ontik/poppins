import os
import asyncio
import json
import base64
import wave
import threading
import time
from typing import Optional, Callable, List, Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview")
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "alloy")

class RealtimeChatbot:
    """
    A realtime chatbot with text and audio support using OpenAI's realtime API
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-realtime-preview", voice: str = "alloy"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.voice = voice
        self.connection = None
        self.connection_manager = None
        self.conversation_history = []
        self.is_connected = False
        
        # Handlers for streaming output
        self.text_handler: Optional[Callable[[str], None]] = None
        self.audio_handler: Optional[Callable[[bytes], None]] = None
        self.response_complete_handler: Optional[Callable[[], None]] = None
        
        # Audio settings
        self.audio_format = "pcm16"
        self.sample_rate = 24000
        
        # Response accumulation
        self.current_text_response = ""
        self.current_audio_response = b""
        
    def set_text_handler(self, handler: Callable[[str], None]):
        """Set callback for text streaming chunks"""
        self.text_handler = handler
        
    def set_audio_handler(self, handler: Callable[[bytes], None]):
        """Set callback for audio streaming chunks"""
        self.audio_handler = handler
        
    def set_response_complete_handler(self, handler: Callable[[], None]):
        """Set callback for when response is complete"""
        self.response_complete_handler = handler
        
    async def connect(self, modalities: List[str] = ["text"]):
        """
        Establish connection to OpenAI realtime API
        modalities: list of "text" and/or "audio"
        """
        try:
            # Create and store the connection manager
            self.connection_manager = self.client.beta.realtime.connect(model=self.model)
            self.connection = await self.connection_manager.__aenter__()
            
            # Configure session
            session_config = {
                "modalities": modalities,
                "instructions": (
                    "You are Poppins, a supportive, empathetic assistant for parents. "
                    "Always respond in English and provide calm, emotionally supportive responses. "
                    "Help parents with their concerns in a caring and understanding manner. "
                    "When audio output is requested, always provide both text and audio responses."
                ),
                "voice": self.voice,
                "input_audio_format": self.audio_format,
                "output_audio_format": self.audio_format,
                "input_audio_transcription": {"model": "whisper-1"}
            }
            
            print(f"🔧 Session config: {session_config}")  # Debug session configuration
            
            await self.connection.session.update(session=session_config)
            self.is_connected = True
            print(f"✅ Connected to OpenAI Realtime API with modalities: {modalities}")
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            raise
            
    async def disconnect(self):
        """Close the connection"""
        if self.connection_manager:
            try:
                # Use the proper async context manager exit
                await self.connection_manager.__aexit__(None, None, None)
                self.is_connected = False
                print("✅ Disconnected from OpenAI Realtime API")
            except Exception as e:
                print(f"⚠️ Error during disconnect: {e}")
            finally:
                self.connection = None
                self.connection_manager = None
                
    async def send_text_message(self, text: str, include_audio: bool = False):
        """Send a text message to the assistant"""
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
            
        # Create conversation item with correct content type
        item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}]
        }
        
        # Add to conversation history
        self.conversation_history.append(item)
        
        # Trim history before sending if it gets too long
        # This prevents hitting API token limits during the session
        max_messages_before_trim = 12  # Trim when we have more than 6 pairs
        if len(self.conversation_history) > max_messages_before_trim:
            self.trim_conversation_history(max_pairs=5)
        
        # Send to API
        await self.connection.conversation.item.create(item=item)
        
        # Configure response based on modalities
        response_config = {}
        if include_audio:
            response_config["modalities"] = ["text", "audio"]
            print(f"🔧 Creating response with modalities: {response_config['modalities']}")
        else:
            print("🔧 Creating text-only response")
        
        # Trigger response
        await self.connection.response.create(response=response_config if response_config else None)
        
    async def send_audio_message(self, audio_data: bytes):
        """Send audio message (PCM16 format)"""
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
            
        # Encode audio to base64
        audio_b64 = base64.b64encode(audio_data).decode()
        
        # Create conversation item
        item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "audio": audio_b64}]
        }
        
        # Add to conversation history
        self.conversation_history.append(item)
        
        # Send to API
        await self.connection.conversation.item.create(item=item)
        
        # Trigger response with audio output
        await self.connection.response.create(response={"modalities": ["text", "audio"]})
        
    async def listen_for_responses(self):
        """Listen for and handle streaming responses"""
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
            
        self.current_text_response = ""
        self.current_audio_response = b""
        
        try:
            async for event in self.connection:
                await self._handle_event(event)
                
                # Break when response is complete
                if event.type == "response.done":
                    # Add complete response to conversation history
                    self._save_response_to_history()
                    
                    if self.response_complete_handler:
                        self.response_complete_handler()
                    break
                    
        except Exception as e:
            print(f"❌ Error listening for responses: {e}")
            raise
            
    async def _handle_event(self, event):
        """Handle individual events from the API"""
        
        # Debug: Log all event types
        print(f"🔍 Event received: {event.type}")
        
        # Text streaming events
        if event.type == "response.text.delta":
            chunk = event.delta
            self.current_text_response += chunk
            print(f"📝 Text delta: '{chunk}'")  # Debug text chunks
            if self.text_handler:
                self.text_handler(chunk)
                
        elif event.type == "response.text.done":
            print(f"📝 Text complete. Total: '{self.current_text_response}'")  # Debug text completion
            
        # Audio transcript events (these contain the text when audio is generated)
        elif event.type == "response.audio_transcript.delta":
            chunk = event.delta
            self.current_text_response += chunk
            print(f"📝 Audio transcript delta: '{chunk}'")  # Debug transcript chunks
            if self.text_handler:
                self.text_handler(chunk)
                
        elif event.type == "response.audio_transcript.done":
            print(f"📝 Audio transcript complete. Total: '{self.current_text_response}'")  # Debug transcript completion
            
        # Audio streaming events
        elif event.type == "response.audio.delta":
            # Decode base64 audio chunk
            audio_chunk = base64.b64decode(event.delta)
            self.current_audio_response += audio_chunk
            print(f"🔊 Audio delta: {len(audio_chunk)} bytes")  # Debug audio chunks
            if self.audio_handler:
                self.audio_handler(audio_chunk)
                
        elif event.type == "response.audio.done":
            print(f"🔊 Audio complete. Total: {len(self.current_audio_response)} bytes")  # Debug audio completion
            
        # Input audio transcription
        elif event.type == "conversation.item.input_audio_transcription.completed":
            transcript = event.transcript
            print(f"🎤 Transcription: {transcript}")
            
        # Error handling
        elif event.type == "error":
            print(f"❌ API Error: {event.error}")
            
        # Debug: Log any other event types we might be missing
        elif hasattr(event, 'type'):
            print(f"🔍 Unhandled event type: {event.type}")
            
    def _save_response_to_history(self):
        """Save the complete response to conversation history"""
        response_content = []
        
        if self.current_text_response:
            response_content.append({"type": "text", "text": self.current_text_response})
            
        if self.current_audio_response:
            audio_b64 = base64.b64encode(self.current_audio_response).decode()
            response_content.append({"type": "audio", "audio": audio_b64})
            
        if response_content:
            assistant_item = {
                "type": "message",
                "role": "assistant",
                "content": response_content
            }
            self.conversation_history.append(assistant_item)
            
        # Keep only the last 5 conversation pairs (10 messages total)
        # This prevents context length from exceeding max token limits
        max_messages = 10  # 5 pairs of user-assistant exchanges
        if len(self.conversation_history) > max_messages:
            # Remove oldest messages while keeping the conversation balanced
            messages_to_remove = len(self.conversation_history) - max_messages
            self.conversation_history = self.conversation_history[messages_to_remove:]
            print(f"🔄 Trimmed conversation history to last {max_messages // 2} pairs")
            
    def get_conversation_history(self) -> List[Dict]:
        """Get the current conversation history"""
        return self.conversation_history.copy()
        
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        
    def trim_conversation_history(self, max_pairs: int = 5):
        """
        Manually trim conversation history to keep only recent exchanges
        max_pairs: number of user-assistant exchange pairs to keep
        """
        max_messages = max_pairs * 2  # Each pair = user + assistant message
        if len(self.conversation_history) > max_messages:
            messages_to_remove = len(self.conversation_history) - max_messages
            self.conversation_history = self.conversation_history[messages_to_remove:]
            print(f"🔄 Manually trimmed conversation history to last {max_pairs} pairs")
            
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation"""
        return {
            "total_messages": len(self.conversation_history),
            "conversation_pairs": len(self.conversation_history) // 2,
            "user_messages": len([msg for msg in self.conversation_history if msg.get("role") == "user"]),
            "assistant_messages": len([msg for msg in self.conversation_history if msg.get("role") == "assistant"])
        }
        
    def save_audio_to_file(self, audio_data: bytes, filename: str):
        """Save audio data to WAV file"""
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)
        print(f"💾 Audio saved to {filename}")

# Test Functions
async def test_text_only():
    """Test text-only conversation"""
    print("\n🧪 Testing Text-Only Conversation")
    print("=" * 50)
    
    chatbot = RealtimeChatbot(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, voice=OPENAI_VOICE)
    
    def text_handler(chunk: str):
        print(chunk, end="", flush=True)
        
    def response_complete_handler():
        print("\n")  # New line after response
        
    chatbot.set_text_handler(text_handler)
    chatbot.set_response_complete_handler(response_complete_handler)
    
    try:
        await chatbot.connect(modalities=["text"])
        
        test_messages = [
            "Hello, I'm a new parent and feeling overwhelmed.",
            "My baby won't stop crying and I don't know what to do.",
            "How can I tell if my baby is hungry or just fussy?"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n💬 Test {i}: {message}")
            print("🤖 Poppins: ", end="", flush=True)
            
            await chatbot.send_text_message(message)
            await chatbot.listen_for_responses()
            
            await asyncio.sleep(1)  # Brief pause between messages
            
    finally:
        await chatbot.disconnect()
        
    stats = chatbot.get_conversation_stats()
    print(f"\n📜 Conversation stats: {stats['conversation_pairs']} pairs, {stats['total_messages']} total messages")
    print(f"📊 Kept last {min(5, stats['conversation_pairs'])} conversation pairs in memory")

async def test_text_interactive():
    """Test interactive text conversation"""
    print("\n🧪 Testing Interactive Text Conversation")
    print("=" * 50)
    print("Type your messages (or 'quit' to exit)")
    
    chatbot = RealtimeChatbot(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, voice=OPENAI_VOICE)
    
    def text_handler(chunk: str):
        print(chunk, end="", flush=True)
        
    def response_complete_handler():
        print("\n")  # New line after response
        
    chatbot.set_text_handler(text_handler)
    chatbot.set_response_complete_handler(response_complete_handler)
    
    try:
        await chatbot.connect(modalities=["text"])
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            print("🤖 Poppins: ", end="", flush=True)
            await chatbot.send_text_message(user_input)
            await chatbot.listen_for_responses()
            
    except KeyboardInterrupt:
        print("\n👋 Conversation ended by user")
    finally:
        await chatbot.disconnect()

async def test_multimodal_placeholder():
    """Test multimodal setup (placeholder for audio implementation)"""
    print("\n🧪 Testing Multimodal Setup")
    print("=" * 50)
    
    chatbot = RealtimeChatbot(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, voice=OPENAI_VOICE)
    
    def text_handler(chunk: str):
        print(chunk, end="", flush=True)
        
    def audio_handler(audio_chunk: bytes):
        print(f"🔊[Audio chunk: {len(audio_chunk)} bytes]", end="", flush=True)
        
    def response_complete_handler():
        print("\n")
        
    chatbot.set_text_handler(text_handler)
    chatbot.set_audio_handler(audio_handler)
    chatbot.set_response_complete_handler(response_complete_handler)
    
    try:
        await chatbot.connect(modalities=["text", "audio"])
        
        print("💬 Sending text message to multimodal session...")
        print("🤖 Poppins: ", end="", flush=True)
        
        await chatbot.send_text_message(
            "Hello! Can you give me a brief, encouraging message for new parents?"
        )
        await chatbot.listen_for_responses()
        
        # Save any audio response
        if chatbot.current_audio_response:
            chatbot.save_audio_to_file(chatbot.current_audio_response, "test_response.wav")
            
    finally:
        await chatbot.disconnect()

async def run_performance_test():
    """Test response times and connection stability"""
    print("\n🧪 Performance Test")
    print("=" * 50)
    
    chatbot = RealtimeChatbot(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, voice=OPENAI_VOICE)
    
    response_times = []
    
    def text_handler(chunk: str):
        pass  # Silent for performance test
        
    chatbot.set_text_handler(text_handler)
    
    try:
        await chatbot.connect(modalities=["text"])
        
        test_messages = [
            "Quick response test 1",
            "Quick response test 2", 
            "Quick response test 3"
        ]
        
        for i, message in enumerate(test_messages, 1):
            start_time = time.time()
            
            await chatbot.send_text_message(message)
            await chatbot.listen_for_responses()
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            print(f"⏱️ Test {i}: {response_time:.2f}s")
            
        avg_time = sum(response_times) / len(response_times)
        print(f"📊 Average response time: {avg_time:.2f}s")
        
    finally:
        await chatbot.disconnect()

async def main():
    """Main function to run all tests"""
    print("🚀 Poppins Realtime Chatbot - Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Basic text conversation
        await test_text_only()
        
        # Test 2: Performance test
        await run_performance_test()
        
        # Test 3: Multimodal setup
        await test_multimodal_placeholder()
        
        # Test 4: Interactive mode (comment out for automated testing)
        choice = input("\n❓ Run interactive test? (y/n): ").strip().lower()
        if choice == 'y':
            await test_text_interactive()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())