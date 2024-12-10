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
load_dotenv()

agent_threads = []

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
if 'current_agent_input' not in st.session_state:
    st.session_state.current_agent_input = []
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'mlflow_launched' not in st.session_state:
    st.session_state.mlflow_launched = False

# Configure LM
lm = dspy.LM(
    model=f"azure/{os.getenv('AZURE_DEPLOYMENT_MODEL')}",
    api_key=os.getenv('AZURE_API_KEY'),
    api_base=os.getenv('AZURE_API_BASE'),
    api_version=os.getenv('AZURE_API_VERSION'),
    cache=False
)
dspy.configure(lm=lm, callbacks=[AgentLoggingCallback()])

agent = AssistantAgent()

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
    col1, col2 = st.columns([1, 4])
    with col1:
        speaker_id = st.text_input(
            "📢 Speaker ID:", 
            value="2", 
            help="Enter the speaker ID (e.g., 1 for client, 2 for advisor)"
        )
    with col2:
        transcribed_text = st.text_area(
            "📝 Enter or paste text:", 
            value=st.session_state['final_transcription'],
            height=150
        )


def add_result(prediction, input_text):
    """Add a new result to the session state list"""
    if 'results_list' not in st.session_state:
        st.session_state.results_list = []
        st.session_state.containers = []
    
    # Create a new container for the new result
    st.session_state.containers.insert(0, st.container())
    
    # Add new result to the beginning of the list
    st.session_state.results_list.insert(0, {
        'prediction': prediction,
        'input_text': input_text,
        'container': st.session_state.containers[0]
    })

def display_single_result(result):
    """Display a single result in its container"""
    with result['container']:
        st.markdown("## Analysis")
        
        # Display assistant results
        st.subheader("🤖 Assistant Results")
        st.write(result['prediction'].relevant_information)
        
        # Add citations section
        st.subheader("📚 References")
        st.write(result['prediction'].citations)
        
        with st.expander("💬 View Agent Input", expanded=False):
            st.write(result['input_text'])

        # Add expandable sections for trajectory and reasoning
        with st.expander("💭 View Reasoning", expanded=False):
            st.write(result['prediction'].reasoning)
        
        with st.expander("🔍 View Trajectory", expanded=False):
            st.write(result['prediction'].trajectory)
        
        st.markdown("---")

def display_results():
    """Display all results from the session state list"""
    if 'results_list' in st.session_state:
        for result in st.session_state.results_list:
            display_single_result(result)

def transcriber_callback(transcription):
    # Create a sidebar for live transcription if it doesn't exist
    if 'live_transcription_container' not in st.session_state:
        st.session_state.live_transcription_container = st.sidebar.empty()
    
    # Handle different transcription types
    if transcription['type'] == 'interim':
        # Update the live transcription display with interim results
        st.session_state.live_transcription = f"Speaker {transcription['speaker_id']}: {transcription['text']}"
        with st.session_state.live_transcription_container:
            st.text(f"""
                * 📝 Live Transcription *
                
                {st.session_state.final_transcription}
                {st.session_state.live_transcription}
            """)
    
    elif transcription['type'] == 'final':
        def run_agent(text):
            lm = dspy.LM(
                model=f"azure/{os.getenv('AZURE_DEPLOYMENT_MODEL')}",
                api_key=os.getenv('AZURE_API_KEY'),
                api_base=os.getenv('AZURE_API_BASE'),
                api_version=os.getenv('AZURE_API_VERSION'),
                cache=False
            )
            dspy.configure(lm=lm, callbacks=[AgentLoggingCallback()])
            agent = AssistantAgent()
            mlflow.dspy.autolog()
            mlflow.set_experiment("Agent Assistant Bank Call - From Audio")
            prediction = agent(transcribed_text=text)
            print(f"got prediction: {prediction.relevant_information}")
            if prediction.relevant_information != "Waiting for more information":
                print(f"calling agent with {text}")
                st.session_state.current_agent_input.append(text)
                st.session_state.prediction_results.append(prediction)
                # Add the new result and display it immediately
                add_result(prediction, text)
                display_single_result(st.session_state.results_list[0])
            
            st.session_state.analysis_complete = True

        # Concatenate final transcription and update display
        new_final = f"Speaker {transcription['speaker_id']}: {transcription['text']}\n"
        st.session_state.final_transcription += new_final
        st.session_state.live_transcription = ""  # Clear interim transcription
        
        thread = threading.Thread(target=run_agent, args=(transcription['text'],))
        add_script_run_ctx(thread)
        thread.start()
        agent_threads.append(thread)


# Create the submit button
if st.button("🤖 Analyze"):
    st.session_state.thought_container = st.empty()

    if input_method == "Write or paste text" and transcribed_text:
        # Start experiment
        mlflow.dspy.autolog()
        mlflow.set_experiment("Agent Assistant Bank Call - From Text")

        # Get prediction
        st.session_state.current_agent_input.append(transcribed_text)
        prediction = agent(transcribed_text=transcribed_text)
        st.session_state.prediction_results.append(prediction)
        
        # Add and display the result immediately
        add_result(prediction, transcribed_text)
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
if st.session_state.analysis_complete and st.session_state.prediction_results:
    # Clear the thought container after agent is done
    if 'thought_container' in st.session_state:
        st.session_state.thought_container.empty()
    
    if input_method == "Write or paste text":
        st.markdown("# Analysis Results")
    
    # MLflow button with callback
    if st.button("📊 View MLflow Experiment Results", on_click=launch_mlflow):
        st.success("🚀 MLflow UI launched! Opening in new tab...")
