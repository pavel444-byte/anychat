from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os
import requests
import json
import atexit
import shutil
from typing import List, Dict, Optional
import threading
import time

# Try to import speech recognition, handle gracefully if not available
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

# Check for audio system availability
def check_audio_system():
    """Check if audio system is available and working"""
    if not SPEECH_RECOGNITION_AVAILABLE or sr is None:
        return False, "Speech recognition library not available"
    
    try:
        # First check if any microphones are available
        mic_list = sr.Microphone.list_microphone_names()
        if not mic_list:
            return False, "No microphone devices found on system"
        
        # Try to initialize recognizer and microphone
        r = sr.Recognizer()
        with sr.Microphone() as source:
            # Quick test - just initialize, don't record
            pass
        return True, f"Audio system available ({len(mic_list)} microphone(s) detected)"
    except OSError as e:
        error_msg = str(e).lower()
        if "alsa" in error_msg or "card" in error_msg:
            return False, "No sound card found - audio hardware may not be available"
        elif "device" in error_msg:
            return False, "Audio device access error - check permissions or hardware"
        return False, f"Audio system error: {str(e)}"
    except Exception as e:
        return False, f"Audio system error: {str(e)}"

def get_microphone_info():
    """Get detailed microphone information for debugging"""
    if not SPEECH_RECOGNITION_AVAILABLE or sr is None:
        return "Speech recognition not available"
    
    try:
        mic_list = sr.Microphone.list_microphone_names()
        if not mic_list:
            return "No microphones detected"
        
        info = f"Found {len(mic_list)} microphone(s):\n"
        for i, name in enumerate(mic_list):
            info += f"  {i}: {name}\n"
        return info.strip()
    except Exception as e:
        return f"Error getting microphone info: {str(e)}"

# Check audio system availability at startup
AUDIO_SYSTEM_AVAILABLE, AUDIO_SYSTEM_ERROR = check_audio_system()

load_dotenv()

# Function to clear all Streamlit cache on exit
def clear_streamlit_cache_on_exit():
    """Clear all Streamlit cache and session data when the app exits"""
    try:
        # Clear Streamlit's cache data
        st.cache_data.clear()
        
        # Clear cache resource if available
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
        
        # Remove Streamlit cache directory
        streamlit_cache_dir = os.path.expanduser("~/.streamlit")
        if os.path.exists(streamlit_cache_dir):
            shutil.rmtree(streamlit_cache_dir, ignore_errors=True)
        
        print("✅ Streamlit cache cleared on exit")
    except Exception as e:
        print(f"⚠️ Warning: Could not clear cache on exit: {e}")

# Register the exit handler
atexit.register(clear_streamlit_cache_on_exit)

# Speech recognition functions
def record_audio():
    """Record audio from microphone and return the recognized text"""
    if not SPEECH_RECOGNITION_AVAILABLE or sr is None:
        return None, "Speech recognition is not available. Please install: pip install speechrecognition pyaudio"
    
    if not AUDIO_SYSTEM_AVAILABLE:
        return None, f"Audio system not available: {AUDIO_SYSTEM_ERROR}"
    
    try:
        # Initialize recognizer
        r = sr.Recognizer()
        
        # Use microphone as source
        with sr.Microphone() as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=1)
            
            # Record audio
            audio = r.listen(source, timeout=10, phrase_time_limit=30)
        
        # Recognize speech using Google's speech recognition
        text = r.recognize_google(audio)  # type: ignore
        return text, None
        
    except sr.RequestError as e:
        return None, f"Could not request results from speech recognition service: {e}"
    except sr.UnknownValueError:
        return None, "Could not understand audio. Please try again."
    except sr.WaitTimeoutError:
        return None, "Listening timeout. Please try again."
    except OSError as e:
        if "ALSA" in str(e) or "card" in str(e).lower():
            return None, "Audio system error: No sound card available (ALSA error)"
        return None, f"Audio system error: {e}"
    except Exception as e:
        return None, f"Error during speech recognition: {e}"

def record_audio_async():
    """Async wrapper for recording audio"""
    if 'recording_status' not in st.session_state:
        st.session_state.recording_status = 'idle'
    
    if 'recorded_text' not in st.session_state:
        st.session_state.recorded_text = ""
    
    if 'recording_error' not in st.session_state:
        st.session_state.recording_error = None
    
    def recording_thread():
        try:
            st.session_state.recording_status = 'recording'
            text, error = record_audio()
            
            if error:
                st.session_state.recording_error = error
                st.session_state.recording_status = 'error'
            else:
                st.session_state.recorded_text = text
                st.session_state.recording_status = 'completed'
        except Exception as e:
            st.session_state.recording_error = f"Recording thread error: {str(e)}"
            st.session_state.recording_status = 'error'
    
    if st.session_state.recording_status == 'idle':
        # Check if audio system is available before starting thread
        if not AUDIO_SYSTEM_AVAILABLE:
            st.session_state.recording_error = AUDIO_SYSTEM_ERROR
            st.session_state.recording_status = 'error'
            return
        
        # Start recording in a separate thread
        thread = threading.Thread(target=recording_thread, name='recording_thread')
        thread.daemon = True
        thread.start()

# Cache for models to avoid repeated API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_openrouter_models(api_key: str) -> List[Dict]:
    """
    Fetch available models from OpenRouter API
    Returns a list of model dictionaries with id, name, and other metadata
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            
            # Sort models by name for better UX
            models.sort(key=lambda x: x.get("name", x.get("id", "")))
            
            return models
        else:
            st.error(f"Failed to fetch models: HTTP {response.status_code}")
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching models: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []

def get_model_options(api_key: str) -> tuple[List[str], List[str]]:
    """
    Get model options for the selectbox
    Returns (model_ids, model_display_names)
    """
    if not api_key:
        # Fallback to popular models if no API key
        fallback_models = [
            "openrouter/horizon-beta",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-haiku",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-3.5-turbo",
            "google/gemini-pro",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "mistralai/mistral-7b-instruct",
            "cohere/command-r-plus",
            "perplexity/llama-3.1-sonar-large-128k-online"
        ]
        return fallback_models, fallback_models
    
    models = fetch_openrouter_models(api_key)
    
    if not models:
        # Fallback if API call fails
        fallback_models = [
            "openrouter/horizon-beta",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "openai/gpt-4o-mini"
        ]
        return fallback_models, fallback_models
    
    model_ids = []
    model_display_names = []
    
    for model in models:
        model_id = model.get("id", "")
        model_name = model.get("name", model_id)
        
        if model_id:
            model_ids.append(model_id)
            # Create a more readable display name
            display_name = f"{model_name} ({model_id})"
            model_display_names.append(display_name)
    
    return model_ids, model_display_names

# Initialize session state for configuration
if "current_model" not in st.session_state:
    st.session_state.current_model = os.getenv("anychat_model") or "openrouter/horizon-beta"

if "current_key" not in st.session_state:
    st.session_state.current_key = os.getenv("anychat_key") or ""

# Initialize session state for cached models
if "cached_models" not in st.session_state:
    st.session_state.cached_models = None

if "models_last_fetched" not in st.session_state:
    st.session_state.models_last_fetched = None

# Streamlit page configuration
st.set_page_config(
    page_title="AnyChat - OpenRouter LLM Chat",
    page_icon="💬",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Enhanced sidebar with model and key editing (moved before main content)
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Model Selection Section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🤖 Model Selection")
    with col2:
        if st.button("🔄", help="Refresh models", key="refresh_models"):
            # Clear cached models to force refresh
            st.session_state.cached_models = None
            st.session_state.models_last_fetched = None
            st.cache_data.clear()
            st.rerun()
    
    # Show loading message while fetching models
    if st.session_state.current_key:
        with st.spinner("Loading available models..."):
            model_ids, model_display_names = get_model_options(st.session_state.current_key)
    else:
        st.warning("⚠️ Enter API key to load available models")
        model_ids, model_display_names = get_model_options("")
    
    # Find current model index
    current_model_index = 0
    if st.session_state.current_model in model_ids:
        current_model_index = model_ids.index(st.session_state.current_model)
    
    # Model dropdown with display names
    if model_ids:
        selected_index = st.selectbox(
            "Choose Model:",
            options=range(len(model_display_names)),
            format_func=lambda x: model_display_names[x],
            index=current_model_index,
            key="model_selector",
            help=f"Found {len(model_ids)} available models"
        )
        
        # Get the actual model ID from the selected index
        selected_model = model_ids[selected_index]
    else:
        st.error("❌ No models available. Check your API key.")
        selected_model = "openrouter/horizon-beta"
    
    # Custom model input
    custom_model = st.text_input(
        "Or enter custom model:",
        placeholder="e.g., custom/model-name",
        key="custom_model_input"
    )
    
    # Use custom model if provided, otherwise use selected
    final_model = custom_model if custom_model.strip() else selected_model
    
    # Update model if changed
    if final_model != st.session_state.current_model:
        st.session_state.current_model = final_model
        st.rerun()
    
    st.divider()
    
    # API Key Section
    st.subheader("🔑 API Key Management")
    
    # API key input
    new_key = st.text_input(
        "OpenRouter API Key:",
        value=st.session_state.current_key,
        type="password",
        help="Your OpenRouter API key (starts with sk-or-v1-)",
        key="api_key_input"
    )
    
    # Update key if changed
    if new_key != st.session_state.current_key:
        st.session_state.current_key = new_key
    
    # Save configuration button
    if st.button("💾 Save Configuration", type="primary"):
        try:
            # Update .env file
            with open('.env', 'w') as f:
                f.write(f"anychat_key={st.session_state.current_key}\n")
                f.write(f"anychat_model={st.session_state.current_model}\n")
            st.success("✅ Configuration saved to .env file!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error saving configuration: {str(e)}")
    
    st.divider()
    
    # Chat Controls Section
    st.subheader("💬 Chat Controls")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Clear all cache button
    if st.button("🧹 Clear All Cache", help="Clear Streamlit cache and session data"):
        try:
            # Clear all Streamlit caches
            st.cache_data.clear()
            if hasattr(st, 'cache_resource'):
                st.cache_resource.clear()
            
            # Clear session state (except essential keys)
            essential_keys = ['current_model', 'current_key']
            keys_to_remove = [key for key in st.session_state.keys() if key not in essential_keys]
            for key in keys_to_remove:
                del st.session_state[key]
            
            st.success("✅ All cache cleared successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error clearing cache: {str(e)}")
    
    st.divider()
    
    # Status Information
    st.subheader("📊 Status")
    
    # Current configuration display
    st.info(f"**Current Model:** {st.session_state.current_model}")
    st.info(f"**Messages:** {len(st.session_state.messages)}")
    
    # API key status
    if st.session_state.current_key:
        # Mask the key for display
        masked_key = st.session_state.current_key[:8] + "..." + st.session_state.current_key[-4:] if len(st.session_state.current_key) > 12 else "***"
        st.success(f"✅ API Key: {masked_key}")
    else:
        st.error("❌ API Key Missing")
        st.warning("Please enter your OpenRouter API key above")
    
    st.divider()
    
    # Audio System Diagnostics
    st.subheader("🎤 Audio System")
    
    if AUDIO_SYSTEM_AVAILABLE:
        st.success(f"✅ {AUDIO_SYSTEM_ERROR}")
    else:
        st.error(f"❌ {AUDIO_SYSTEM_ERROR}")
    
    # Show microphone details in an expander
    with st.expander("🔍 Audio Diagnostics", expanded=False):
        mic_info = get_microphone_info()
        st.text(mic_info)
        
        if not AUDIO_SYSTEM_AVAILABLE:
            st.markdown("""
            **Troubleshooting Audio Issues:**
            
            1. **No sound card found**: Your system may be running in a container or virtual environment without audio hardware
            2. **Check hardware**: Ensure a microphone is connected and recognized by your system
            3. **Install audio drivers**: You may need to install or configure audio drivers
            4. **Container/VM**: If running in Docker/VM, you may need to enable audio passthrough
            5. **Permissions**: Check if your user has permission to access audio devices
            
            **For Linux systems:**
            ```bash
            # Check for audio devices
            arecord -l
            
            # Check if user is in audio group
            groups $USER
            
            # Add user to audio group if needed
            sudo usermod -a -G audio $USER
            ```
            """)

# Configuration (moved after sidebar)
user_key = st.session_state.current_key
model = st.session_state.current_model

# App header
st.title("💬 AnyChat")
st.caption(f"Chat with **{model}** via OpenRouter")

# Validate configuration and show appropriate content
if not user_key:
    st.error("❌ API Key not found. Please enter your API key in the sidebar.")
    st.info("👈 Use the sidebar on the left to configure your OpenRouter API key and select a model.")
    st.stop()

# Initialize OpenAI client with OpenRouter
client = OpenAI(api_key=user_key, base_url="https://openrouter.ai/api/v1")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize session state for microphone
if 'recording_status' not in st.session_state:
    st.session_state.recording_status = 'idle'
if 'recorded_text' not in st.session_state:
    st.session_state.recorded_text = ""
if 'recording_error' not in st.session_state:
    st.session_state.recording_error = None

# Chat input with microphone button
col1, col2 = st.columns([6, 1])

with col1:
    # Regular text input
    prompt = st.chat_input("What would you like to chat about?")

with col2:
    # Microphone button - only show if audio system is available
    if AUDIO_SYSTEM_AVAILABLE:
        if st.session_state.recording_status == 'idle':
            if st.button("🎤", help="Click to record voice message", key="mic_button"):
                record_audio_async()
                st.rerun()
        elif st.session_state.recording_status == 'recording':
            st.button("🔴 Recording...", disabled=True, key="recording_button")
            # Auto-refresh to check recording status
            time.sleep(0.1)
            st.rerun()
        elif st.session_state.recording_status == 'completed':
            if st.button("✅ Use Recording", help="Click to use the recorded text", key="use_recording_button"):
                prompt = st.session_state.recorded_text
                # Reset recording state
                st.session_state.recording_status = 'idle'
                st.session_state.recorded_text = ""
                st.session_state.recording_error = None
        elif st.session_state.recording_status == 'error':
            if st.button("❌ Try Again", help="Recording failed, click to try again", key="retry_button"):
                st.session_state.recording_status = 'idle'
                st.session_state.recording_error = None
                st.rerun()
    else:
        # Show disabled microphone button with tooltip explaining why it's disabled
        st.button("🎤", disabled=True, help=f"Voice input disabled: {AUDIO_SYSTEM_ERROR}", key="mic_disabled")

# Show recording status and text
if st.session_state.recording_status == 'recording':
    st.info("🎤 Listening... Speak now!")
elif st.session_state.recording_status == 'completed' and st.session_state.recorded_text:
    st.success(f"🎤 Recorded: \"{st.session_state.recorded_text}\"")
elif st.session_state.recording_status == 'error' and st.session_state.recording_error:
    st.error(f"🎤 {st.session_state.recording_error}")

# Process the prompt (from text input or voice)
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        try:
            # Create a placeholder for streaming response
            message_placeholder = st.empty()
            full_response = ""
            
            # Make API call to OpenRouter
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Stream the response
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            # Final response without cursor
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            full_response = error_message
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Reset recording state if it was from voice input
    if st.session_state.recording_status == 'completed':
        st.session_state.recording_status = 'idle'
        st.session_state.recorded_text = ""
        st.session_state.recording_error = None
