import streamlit as st
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import pypdf
from pptx import Presentation
import io

# Load environment variables
load_dotenv()
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")

# Initialize Gemini Client for Vertex AI
@st.cache_resource
def get_gemini_client():
    if not PROJECT_ID or not LOCATION:
        return genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None

client = get_gemini_client()

st.set_page_config(page_title="Learning Assistant", layout="wide")
st.title("📚 Learning Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Materials")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, JPG, PNG, MP3, WAV, PPTX)",
        type=["pdf", "jpg", "jpeg", "png", "mp3", "wav", "pptx"],
        accept_multiple_files=True
    )

def process_file(uploaded_file):
    """Process an uploaded file into a Gemini Part or text."""
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name
    mime_type = uploaded_file.type
    
    # Process images and audio as parts
    if mime_type.startswith("image/") or mime_type.startswith("audio/"):
        return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    
    # Process PPTX
    elif file_name.endswith(".pptx"):
        try:
            prs = Presentation(io.BytesIO(file_bytes))
            text_content = []
            for i, slide in enumerate(prs.slides):
                text_content.append(f"--- Slide {i+1} ---")
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text_content.append(shape.text)
            return "\n".join(text_content)
        except Exception as e:
            st.error(f"Error reading PPTX {file_name}: {e}")
            return None

    # Process PDF
    elif file_name.endswith(".pdf"):
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text_content = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_content.append(f"--- Page {i+1} ---")
                    text_content.append(text)
            return "\n".join(text_content)
        except Exception as e:
            st.error(f"Error reading PDF {file_name}: {e}")
            return None
            
    return None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your materials..."):
    # Append user prompt to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not client:
        st.error("Gemini client is not initialized.")
    else:
        # Prepare contents for the model
        contents = []
        
        # Add past messages as context (as text)
        for msg in st.session_state.messages[:-1]:
            role = "USER" if msg["role"] == "user" else "MODEL"
            contents.append(f"{role}: {msg['content']}")
            
        current_turn_parts = [prompt]
        
        # Add files to the current turn
        if uploaded_files:
            for file in uploaded_files:
                processed_part = process_file(file)
                if processed_part:
                    current_turn_parts.append(processed_part)
        
        # Combine history context and current parts
        if contents:
            history_text = "Previous conversation context:\n" + "\n".join(contents) + "\n\nCurrent request:\n"
            current_turn_parts.insert(0, history_text)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                # Using gemini-2.5-flash as the default model
                response_stream = client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=current_turn_parts
                )
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                # Append assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
