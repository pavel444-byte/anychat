from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os
import requests
import json
import atexit
import shutil
from typing import List, Dict, Optional


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

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = os.getenv("anychat_system_prompt") or ""

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
    
    # System Prompt Section
    st.subheader("📝 System Prompt")
    
    # System prompt text area
    system_prompt = st.text_area(
        "Enter system prompt:",
        value=st.session_state.system_prompt,
        height=150,
        placeholder="You are a helpful AI assistant. Be concise and accurate in your responses.",
        help="This prompt will be sent as the system message to guide the AI's behavior",
        key="system_prompt_input"
    )
    
    # Update system prompt if changed
    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
    
    # System prompt controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Prompt", help="Save system prompt to .env file"):
            try:
                # Read existing .env content
                env_content = {}
                if os.path.exists('.env'):
                    with open('.env', 'r') as f:
                        for line in f:
                            if '=' in line and not line.strip().startswith('#'):
                                key, value = line.strip().split('=', 1)
                                env_content[key] = value
                
                # Update with system prompt
                env_content['anychat_system_prompt'] = st.session_state.system_prompt
                env_content['anychat_key'] = st.session_state.current_key
                env_content['anychat_model'] = st.session_state.current_model
                
                # Write back to .env
                with open('.env', 'w') as f:
                    for key, value in env_content.items():
                        f.write(f"{key}={value}\n")
                
                st.success("✅ System prompt saved!")
            except Exception as e:
                st.error(f"❌ Error saving prompt: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear Prompt", help="Clear the system prompt"):
            st.session_state.system_prompt = ""
            st.rerun()
    
    st.divider()
    
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
                f.write(f"anychat_system_prompt={st.session_state.system_prompt}\n")
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
    
    # System prompt status
    if st.session_state.system_prompt.strip():
        prompt_preview = st.session_state.system_prompt[:50] + "..." if len(st.session_state.system_prompt) > 50 else st.session_state.system_prompt
        st.success(f"✅ System Prompt: {prompt_preview}")
    else:
        st.info("ℹ️ No system prompt set")
    
    # API key status
    if st.session_state.current_key:
        # Mask the key for display
        masked_key = st.session_state.current_key[:8] + "..." + st.session_state.current_key[-4:] if len(st.session_state.current_key) > 12 else "***"
        st.success(f"✅ API Key: {masked_key}")
    else:
        st.error("❌ API Key Missing")
        st.warning("Please enter your OpenRouter API key above")
    

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


# Chat input
prompt = st.chat_input("What would you like to chat about?")

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
            
            # Prepare messages with system prompt if provided
            messages = []
            if st.session_state.system_prompt.strip():
                messages.append({"role": "system", "content": st.session_state.system_prompt})
            
            # Add chat history
            messages.extend([
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ])
            
            # Make API call to OpenRouter
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
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
    
