import streamlit as st
from bank_call_agent import AssistantAgent
import dspy
import mlflow
import subprocess
import webbrowser
import time
import threading
from dotenv import load_dotenv
import os
import dspy
from dspy.utils.callback import BaseCallback
import json
load_dotenv()

# At the top of the file, after the imports, initialize session state for live transcription
if 'live_transcription' not in st.session_state:
    st.session_state.live_transcription = ""
if 'final_transcription' not in st.session_state:
    st.session_state.final_transcription = ""

def transcriber_callback(transcription):
    # Create a sidebar for live transcription if it doesn't exist
    if 'live_transcription_container' not in st.session_state:
        st.session_state.live_transcription_container = st.sidebar.empty()
    
    # Handle different transcription types
    if transcription['type'] == 'interim':
        # Update the live transcription display with interim results
        st.session_state.live_transcription = f"Speaker {transcription['speaker_id']}: {transcription['text']}"
        st.session_state.live_transcription_container.markdown(f"""
        ### 📝 Live Transcription
        {st.session_state.final_transcription}
        *{st.session_state.live_transcription}*
        """)
    
    elif transcription['type'] == 'final':
        # Concatenate final transcription and update display
        new_final = f"Speaker {transcription['speaker_id']}: {transcription['text']}\n"
        st.session_state.final_transcription += new_final
        st.session_state.live_transcription = ""  # Clear interim transcription

# Custom callback for displaying thoughts and actions
class AgentLoggingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        if 'thought_container' not in st.session_state:
            st.session_state.thought_container = st.empty()
    
    def on_module_end(self, call_id, outputs, exception):
        # Update the display with current step
        with st.session_state.thought_container.container():
            st.subheader("🤔 Agent's Current Step")
            if "next_thought" in outputs:
                st.markdown(f"**💭 Thinking:** {outputs['next_thought']}")
            
            if "next_tool_name" in outputs:
                if outputs["next_tool_name"].lower() == "finish":
                    st.markdown("**✅ Finish**")
                else:
                    args_str = json.dumps(outputs.get("next_tool_args", {}), indent=2)
                    st.markdown(f"**🔧 Using Tool:** calling `{outputs['next_tool_name']}` with `{args_str}`")

# Configure LM
lm = dspy.LM(
    model=f"azure/{os.getenv('AZURE_DEPLOYMENT_MODEL')}",
    api_key=os.getenv('AZURE_API_KEY'),
    api_base=os.getenv('AZURE_API_BASE'),
    api_version=os.getenv('AZURE_API_VERSION'),
    cache=False
)
dspy.configure(lm=lm, callbacks=[AgentLoggingCallback()])

st.title("Call Assistant 📳 🤖")

# Add file uploader for audio
uploaded_file = st.file_uploader("📂 Upload an audio file", type=['wav'])

# Add transcribe button and handle transcription
if uploaded_file is not None:
    # Create two columns for audio player and transcribe button
    audio_col, button_col = st.columns([3, 1])
    
    with audio_col:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
    
    with button_col:
        # Center the button vertically with some padding
        st.write("")  # Add some vertical spacing
        transcribe_button = st.button("📝🔊 Transcribe")
    
    # Save the uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    if transcribe_button:
        with st.spinner("🤖 Transcribing audio..."):
            from stt import recognize_from_file, transcription_manager
            transcription_manager.set_consumer_callback(transcriber_callback)
            recognize_from_file("temp_audio.wav")
            st.success("✨ Transcription complete!")
        
        # Clean up the temporary file
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")


# Initialize session states
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'mlflow_launched' not in st.session_state:
    st.session_state.mlflow_launched = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

agent = AssistantAgent()

# Create the input text area
col1, col2 = st.columns([1, 4])
with col1:
    speaker_id = st.text_input("📢 Speaker ID:", value="2", help="Enter the speaker ID (e.g., 1 for client, 2 for advisor)")
with col2:
    transcribed_text = st.text_area("📝 Transcribed text from the call:", 
                                   value=st.session_state['final_transcription'],
                                   height=150)

def launch_mlflow():
    if not st.session_state.mlflow_launched:
        def run_mlflow():
            subprocess.Popen(['mlflow', 'ui', '--port', '5001'])
            time.sleep(2)
            webbrowser.open('http://127.0.0.1:5001', new=2)
        
        thread = threading.Thread(target=run_mlflow)
        thread.start()
        st.session_state.mlflow_launched = True

# Create the submit button
if st.button("🔍 Analyze"):
    if transcribed_text:
        # Clear previous results by emptying the containers
        st.session_state.thought_container = st.empty()
        st.session_state.results_container = st.empty()
        st.session_state.references_container = st.empty()
        
        # Start experiment
        mlflow.dspy.autolog()
        mlflow.set_experiment("Agent Assistant Bank Call")

        # Get prediction
        prediction = agent(transcribed_text=transcribed_text)
        
        # Store results in session state
        st.session_state.prediction_results = prediction
        st.session_state.analysis_complete = True
    else:
        st.warning("Provide some transcribed text from the call.")

# Display results if available
if st.session_state.analysis_complete and st.session_state.prediction_results:
    prediction = st.session_state.prediction_results
    
    # Display results in the designated containers
    with st.session_state.results_container.container():
        st.subheader("Assistant Results")
        st.write(prediction.relevant_information)
    
    # Add citations section
    with st.session_state.references_container.container():
        st.subheader("📚 References:")
        st.write(prediction.citations)
        
        st.markdown("---")
        
        # Add expandable sections for trajectory and reasoning
        with st.expander("💭 View Prediction Reasoning", expanded=False):
            st.write(prediction.reasoning)

        with st.expander("🔍 View Prediction Trajectory", expanded=False):
            st.write(prediction.trajectory)
        
        st.markdown("---")
    
    # MLflow button with callback
    if st.button("📊 View MLflow Experiment Results", on_click=launch_mlflow):
        st.success("🚀 MLflow UI launched! Opening in new tab...")
else:
    if not transcribed_text:
        st.warning("Provide some transcribed text from the call.")
