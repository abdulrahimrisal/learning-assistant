import time
import io
import streamlit as st
from google import genai
from google.genai import types
import pypdf
from pptx import Presentation

# ==========================================
# 1. INITIALIZE GEMINI CLIENT
# ==========================================
@st.cache_resource
def get_gemini_client():
    try:
        # Securely load the API key from Streamlit secrets
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None

client = get_gemini_client()

st.set_page_config(page_title="Learning Assistant", layout="wide")
st.title("📚 Learning Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. SIDEBAR & FILE HANDLING
# ==========================================
with st.sidebar:
    st.header("Upload Materials")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, JPG, PNG, MP3, WAV, PPTX)",
        type=["pdf", "jpg", "jpeg", "png", "mp3", "wav", "pptx"],
        accept_multiple_files=True
    )

# MODIFICATION 1: Cache the heavy text extraction so it only runs once per file
@st.cache_data(show_spinner=False)
def extract_text_from_file(file_bytes, file_name):
    text_content = []
    
    # Process PPTX
    if file_name.endswith(".pptx"):
        try:
            prs = Presentation(io.BytesIO(file_bytes))
            for i, slide in enumerate(prs.slides):
                text_content.append(f"--- Slide {i+1} ---")
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        text_content.append(shape.text)
            return "\n".join(text_content)
        except Exception as e:
            return f"Error reading PPTX: {e}"
            
    # Process PDF
    elif file_name.endswith(".pdf"):
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_content.append(f"--- Page {i+1} ---")
                    text_content.append(text)
            return "\n".join(text_content)
        except Exception as e:
            return f"Error reading PDF: {e}"
            
    return None

def process_file(uploaded_file):
    """Process an uploaded file into a Gemini Part or text."""
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name
    mime_type = uploaded_file.type
    
    # Audio and Images get processed fresh as Parts (GenAI SDK handles these natively)
    if mime_type.startswith("image/") or mime_type.startswith("audio/"):
        return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    
    # Documents use the cached helper function to save massive amounts of time
    elif file_name.endswith((".pdf", ".pptx")):
        extracted_text = extract_text_from_file(file_bytes, file_name)
        if extracted_text:
            return extracted_text
            
    return None

# ==========================================
# 3. CHAT INTERFACE & LOGIC
# ==========================================
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
            
        system_instruction = (
            "You are a university learning assistant. You must answer questions STRICTLY and ONLY based on "
            "the uploaded materials. If a student asks a question outside of the provided context, politely decline."
        )
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # MODIFICATION 2: Added Retry Logic for 503 High Demand errors
            for attempt in range(3):
                try:
                    # MODIFICATION 3: Changed to the stable 2.5-flash model
                    response_stream = client.models.generate_content_stream(
                        model="gemini-3.5-flash-lite",
                        contents=current_turn_parts,
                        config=config
                    )
                    
                    # Stream the text to the UI
                    for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")
                            
                    # Remove the blinking cursor when done
                    message_placeholder.markdown(full_response)
                    
                    # Append assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Break the retry loop on success
                    break 
                    
                except Exception as e:
                    if "503" in str(e) and attempt < 2:
                        # Wait 2 seconds before trying again if the server is busy
                        time.sleep(2)
                    else:
                        st.error(f"Error generating response: {e}")
                        break
