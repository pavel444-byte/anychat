from openai import OpenAI
from anthropic import Anthropic, APIStatusError
import streamlit as st
from dotenv import load_dotenv
import os
import requests
import json
from typing import List, Dict, Optional


load_dotenv()




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
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_openai_models(api_key: str) -> List[Dict]:
    """
    Fetch available models from OpenAI API
    Returns a list of model dictionaries with id and name
    """
    if not api_key:
        return []
    try:
        client = OpenAI(api_key=api_key)
        models_response = client.models.list()
        
        # Filter for chat models and sort
        chat_models = []
        for model in models_response.data:
            # Heuristic to identify chat models: typically start with 'gpt-'
            # and are not older completion models or embeddings models
            if model.id.startswith("gpt-") and "instruct" not in model.id and "embedding" not in model.id:
                chat_models.append({"id": model.id, "name": model.id})
        
        chat_models.sort(key=lambda x: x.get("name", x.get("id", "")))
        return chat_models
    except Exception as e:
        st.error(f"Error fetching OpenAI models: {str(e)}")
        return []

def get_model_options(provider: str, api_key: str) -> tuple[List[str], List[str]]:
    """
    Get model options for the selectbox based on the selected provider.
    Returns (model_ids, model_display_names)
    """
    model_ids = []
    model_display_names = []

    if provider == "OpenRouter":
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
        
        for model in models:
            model_id = model.get("id", "")
            model_name = model.get("name", model_id)
            
            if model_id:
                model_ids.append(model_id)
                # Create a more readable display name
                display_name = f"{model_name} ({model_id})"
                model_display_names.append(display_name)
    
    elif provider == "OpenAI":
        if not api_key:
            # Fallback to popular models if no API key
            fallback_models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "gpt-4-turbo",
                "gpt-4"
            ]
            return fallback_models, fallback_models
        
        models = fetch_openai_models(api_key)
        
        if not models:
            # Fallback if API call fails
            fallback_models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo"
            ]
            return fallback_models, fallback_models
        
        for model in models:
            model_id = model.get("id", "")
            model_name = model.get("name", model_id)
            
            if model_id:
                model_ids.append(model_id)
                display_name = f"{model_name} (OpenAI)"
                model_display_names.append(display_name)
    
    elif provider == "Anthropic":
        # Hardcoded popular Anthropic models
        model_ids = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-instant-1.2"
        ]
        model_display_names = [f"{m} (Anthropic)" for m in model_ids]

    return model_ids, model_display_names

# Initialize session state for configuration
if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = os.getenv("ANYCHAT_PROVIDER") or "OpenRouter"

if "current_model" not in st.session_state:
    st.session_state.current_model = os.getenv("ANYCHAT_MODEL") or "openrouter/horizon-beta"

if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "OpenRouter": os.getenv("ANYCHAT_OPENROUTER_KEY") or "",
        "OpenAI": os.getenv("ANYCHAT_OPENAI_KEY") or "",
        "Anthropic": os.getenv("ANYCHAT_ANTHROPIC_KEY") or "",
    }

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = os.getenv("ANYCHAT_SYSTEM_PROMPT") or ""

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

    # Provider Selection Section
    st.subheader("🌐 Provider Selection")
    providers = ["OpenRouter", "OpenAI", "Anthropic"]
    selected_provider = st.selectbox(
        "Choose Provider:",
        options=providers,
        index=providers.index(st.session_state.selected_provider),
        key="provider_selector",
        help="Select the AI model provider"
    )

    if selected_provider != st.session_state.selected_provider:
        st.session_state.selected_provider = selected_provider
        # Reset model when provider changes
        if selected_provider == "OpenRouter":
            st.session_state.current_model = "openrouter/horizon-beta"
        elif selected_provider == "OpenAI":
            st.session_state.current_model = "gpt-4o"
        elif selected_provider == "Anthropic":
            st.session_state.current_model = "claude-3-opus-20240229"
        st.rerun()

    st.divider()
    
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
                env_content['ANYCHAT_SYSTEM_PROMPT'] = st.session_state.system_prompt
                env_content['ANYCHAT_OPENROUTER_KEY'] = st.session_state.api_keys["OpenRouter"]
                env_content['ANYCHAT_OPENAI_KEY'] = st.session_state.api_keys["OpenAI"]
                env_content['ANYCHAT_ANTHROPIC_KEY'] = st.session_state.api_keys["Anthropic"]
                env_content['ANYCHAT_MODEL'] = st.session_state.current_model
                env_content['ANYCHAT_PROVIDER'] = st.session_state.selected_provider
                
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
    current_api_key = st.session_state.api_keys.get(st.session_state.selected_provider, "")
    if current_api_key or st.session_state.selected_provider != "OpenRouter": # OpenRouter can use hardcoded models without key
        with st.spinner("Loading available models..."):
            model_ids, model_display_names = get_model_options(st.session_state.selected_provider, current_api_key)
    else:
        st.warning(f"⚠️ Enter API key for {st.session_state.selected_provider} to load available models")
        model_ids, model_display_names = get_model_options(st.session_state.selected_provider, "")
    
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
    provider_key_label = f"{st.session_state.selected_provider} API Key:"
    provider_key_help = f"Your {st.session_state.selected_provider} API key"
    if st.session_state.selected_provider == "OpenRouter":
        provider_key_help += " (starts with sk-or-v1-)"
    elif st.session_state.selected_provider == "OpenAI":
        provider_key_help += " (starts with sk-)"
    elif st.session_state.selected_provider == "Anthropic":
        provider_key_help += " (starts with sk-ant-)"

    new_key = st.text_input(
        provider_key_label,
        value=st.session_state.api_keys.get(st.session_state.selected_provider, ""),
        type="password",
        help=provider_key_help,
        key=f"api_key_input_{st.session_state.selected_provider}"
    )
    
    # Update key if changed
    if new_key != st.session_state.api_keys.get(st.session_state.selected_provider, ""):
        st.session_state.api_keys[st.session_state.selected_provider] = new_key
    
    # Save configuration button
    if st.button("💾 Save Configuration", type="primary"):
        try:
            # Update .env file
            with open('.env', 'w') as f:
                f.write(f"ANYCHAT_PROVIDER={st.session_state.selected_provider}\n")
                f.write(f"ANYCHAT_MODEL={st.session_state.current_model}\n")
                f.write(f"ANYCHAT_SYSTEM_PROMPT={st.session_state.system_prompt}\n")
                f.write(f"ANYCHAT_OPENROUTER_KEY={st.session_state.api_keys['OpenRouter']}\n")
                f.write(f"ANYCHAT_OPENAI_KEY={st.session_state.api_keys['OpenAI']}\n")
                f.write(f"ANYCHAT_ANTHROPIC_KEY={st.session_state.api_keys['Anthropic']}\n")
            st.success("✅ Configuration saved to .env file!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error saving configuration: {str(e)}")
    
    st.divider()
    
    # Chat Controls Section
    st.subheader("Controls")
    
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
            essential_keys = ['selected_provider', 'current_model', 'api_keys']
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
    current_api_key_display = st.session_state.api_keys.get(st.session_state.selected_provider, "")
    if current_api_key_display:
        # Mask the key for display
        masked_key = current_api_key_display[:8] + "..." + current_api_key_display[-4:] if len(current_api_key_display) > 12 else "***"
        st.success(f"✅ {st.session_state.selected_provider} API Key: {masked_key}")
    else:
        st.error(f"❌ {st.session_state.selected_provider} API Key Missing")
        st.warning(f"Please enter your {st.session_state.selected_provider} API key above")
    

# Configuration (moved after sidebar)
selected_provider = st.session_state.selected_provider
user_key = st.session_state.api_keys.get(selected_provider, "")
model = st.session_state.current_model

# App header
st.title("💬 AnyChat")
st.caption(f"Chat with **{model}** via {selected_provider}")

# Validate configuration and show appropriate content
if not user_key and selected_provider != "OpenRouter": # OpenRouter can use hardcoded models without key
    st.error(f"❌ {selected_provider} API Key not found. Please enter your API key in the sidebar.")
    st.info(f"👈 Use the sidebar on the left to configure your {selected_provider} API key and select a model.")
    st.stop()

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
            # Initialize client based on selected provider
            client = None
            if selected_provider == "OpenRouter":
                client = OpenAI(api_key=user_key, base_url="https://openrouter.ai/api/v1")
            elif selected_provider == "OpenAI":
                client = OpenAI(api_key=user_key)
            elif selected_provider == "Anthropic":
                client = Anthropic(api_key=user_key)

            if client is None:
                st.error(f"❌ Failed to initialize client for {selected_provider}. Please check your API key and selected provider.")
                st.stop()

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
            
            # Make API call based on selected provider
            if selected_provider in ["OpenRouter", "OpenAI"]:
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
            elif selected_provider == "Anthropic":
                # Anthropic API expects messages in a specific format
                anthropic_messages = []
                system_message = None
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

                with client.messages.stream(
                    model=model,
                    max_tokens=1000,
                    messages=anthropic_messages,
                    system=system_message if system_message else "" # Ensure system is always a string
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        message_placeholder.markdown(full_response + "▌")
            
            # Final response without cursor
            message_placeholder.markdown(full_response)
            
        except APIStatusError as e:
            error_message = f"API Error ({e.status_code}): {e.response}"
            st.error(error_message)
            full_response = error_message
        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            full_response = error_message
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
