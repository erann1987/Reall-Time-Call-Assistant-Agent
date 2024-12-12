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
from streamlit.runtime.scriptrunner import add_script_run_ctx
from datetime import datetime
import yaml
load_dotenv()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

agent_threads = []

# Custom callback for displaying thoughts and actions
class AgentLoggingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
    
    def on_module_end(self, call_id, outputs, exception):
        # Update the display with current step
        with st.session_state.thought_container.container():
            if "next_thought" in outputs:
                st.markdown(f"**💭 Thinking:** {outputs['next_thought']}")
            
            if "next_tool_name" in outputs:
                if outputs["next_tool_name"].lower() == "finish":
                    st.markdown("**✅ Finish**")
                else:
                    args_str = json.dumps(outputs.get("next_tool_args", {}), indent=2)
                    st.markdown(f"**🔧 Using Tool:** `{outputs['next_tool_name']}` with `{args_str}`")

def dspy_configure():
    lm = dspy.LM(
        model=f"azure/{config.get('azure_deployment_model')}",
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_base=os.getenv('AZURE_OPENAI_API_BASE'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        cache=False
    )
    dspy.configure(lm=lm, callbacks=[AgentLoggingCallback()])

def launch_mlflow():
    if not st.session_state.mlflow_launched:
        def run_mlflow():
            subprocess.Popen(['mlflow', 'ui', '--port', '5001'])
            time.sleep(2)
            webbrowser.open('http://127.0.0.1:5001', new=2)
        
        thread = threading.Thread(target=run_mlflow)
        thread.start()
        st.session_state.mlflow_launched = True


# Initialize session states
if 'live_transcription' not in st.session_state:
    st.session_state.live_transcription = ""
if 'final_transcription' not in st.session_state:
    st.session_state.final_transcription = ""
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'mlflow_launched' not in st.session_state:
    st.session_state.mlflow_launched = False
if 'results_list' not in st.session_state:
    st.session_state.results_list = []
if 'thought_container' not in st.session_state:
    st.session_state.thought_container = st.empty()
if 'results_placeholder' not in st.session_state:
    st.session_state.results_placeholder = st.empty()

st.title("Call Assistant 📳 🤖")

# Add input method selection
input_method = st.radio(
    "Choose input method:",
    ["Write or paste text", "Upload audio file"],
    horizontal=True
)
transcribed_text = None
if input_method == "Upload audio file":
    # Add file uploader for audio
    uploaded_file = st.file_uploader("📂 Upload an audio file", type=['wav', 'mp3'])

    # Add transcribe button and handle transcription
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[1]}')
        # Save the uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
else:
    transcribed_text = st.text_area("📝 Enter or paste text:", height=150)


def display_results():
    st.session_state.results_placeholder.empty()
    with st.session_state.results_placeholder.container():
        st.session_state.results_list.sort(key=lambda x: x['timestamp'], reverse=True)
        for result in st.session_state.results_list:
            st.subheader("🤖 Assistant Results")
            st.write(result['prediction'].relevant_information)
            
            st.subheader("📚 References")
            st.write(result['prediction'].citations)
            
            with st.expander("💬 View Agent Input", expanded=False):
                st.write(result['input_text'])

            with st.expander("💭 View Reasoning", expanded=False):
                st.write(result['prediction'].reasoning)
            
            with st.expander("🔍 View Trajectory", expanded=False):
                st.write(result['prediction'].trajectory)
            
            st.markdown("---")


def transcriber_callback(transcription):
    # Create a sidebar for live transcription if it doesn't exist
    if 'live_transcription_container' not in st.session_state:
        st.session_state.live_transcription_container = st.sidebar.empty()
    
    # Handle different transcription types
    if transcription['type'] == 'interim':
        # Update the live transcription display with interim results
        st.session_state.live_transcription = f"Speaker {transcription['speaker_id']}: {transcription['text']}"
        with st.session_state.live_transcription_container:
            st.text(
                f"""* 📝 Live Transcription *
                {st.session_state.final_transcription}
                {st.session_state.live_transcription}"""
            )
    
    elif transcription['type'] == 'final':
        def run_agent(text, timestamp):
            dspy_configure()
            agent = AssistantAgent()
            mlflow.dspy.autolog()
            mlflow.set_experiment("Agent Assistant Bank Call - From Audio")
            prediction = agent(transcribed_text=text)
            
            if prediction.relevant_information != "Waiting for more information":
                print(f"got relevant information")
                st.session_state.results_list.append({
                    'prediction': prediction,
                    'input_text': text,
                    'timestamp': timestamp
                })
                display_results()
            
            st.session_state.analysis_complete = True

        # Concatenate final transcription and update display
        new_final = f"Speaker {transcription['speaker_id']}: {transcription['text']}\n"
        st.session_state.final_transcription += f"\n{new_final}"
        st.session_state.live_transcription = ""  # Clear interim transcription
        
        print(f"got utterance: {transcription['text']}")
        text = f"Speaker {transcription['speaker_id']}: {transcription['text']}"
        thread = threading.Thread(
            target=run_agent, 
            args=(text, datetime.now(),)
        )
        add_script_run_ctx(thread)
        thread.start()
        agent_threads.append(thread)


# Create the submit button
if st.button("🤖 Analyze"):
    st.session_state.thought_container = st.empty()

    if input_method == "Write or paste text" and transcribed_text:
        dspy_configure()
        agent = AssistantAgent()
        mlflow.dspy.autolog()
        mlflow.set_experiment("Agent Assistant Bank Call - From Text")
        prediction = agent(transcribed_text=transcribed_text)
        
        # Add and display the result immediately
        st.session_state.results_list.append({
            'prediction': prediction,
            'input_text': transcribed_text,
            'timestamp': datetime.now()
        })
        display_results()
        
        st.session_state.analysis_complete = True
    elif input_method == "Upload audio file" and uploaded_file:
        with st.spinner("🤖 Analyzing audio..."):
            from stt import recognize_from_file, transcription_manager
            transcription_manager.set_consumer_callback(transcriber_callback)
            recognize_from_file("temp_audio.wav")
            for thread in agent_threads:
                thread.join()
            st.success("✨ Analysis complete!")
        
        # Clean up the temporary file
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
    elif input_method == "Write or paste text":
        st.warning("Provide some transcribed text from the call.")
    else:
        st.warning("Upload an audio file to analyze.")

# Modify the display results section
if st.session_state.analysis_complete:
    # Clear the thought container after agent is done
    if 'thought_container' in st.session_state:
        st.session_state.thought_container.empty()
    
    # MLflow button with callback
    if st.button("📊 View MLflow Experiment Results", on_click=launch_mlflow):
        st.success("🚀 MLflow UI launched! Opening in new tab...")
