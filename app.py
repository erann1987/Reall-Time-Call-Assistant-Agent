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

# Custom callback for displaying thoughts and actions
class AgentLoggingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        if 'thought_container' not in st.session_state:
            st.session_state.thought_container = st.empty()
    
    def on_module_start(self, call_id, inputs):
        # Clear previous display
        st.session_state.thought_container.empty()
    
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

st.title("Bank Call Assistant")

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
    speaker_id = st.text_input("Speaker ID:", value="2", help="Enter the speaker ID (e.g., 1 for client, 2 for advisor)")
with col2:
    utterance = st.text_area("Enter the call utterance:", height=150)

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
if st.button("Analyze Utterance"):
    if utterance:
        # st.subheader("🤔 Agent's Current Step")
        # Create thought container here
        st.session_state.thought_container = st.empty()
        
        # Start experiment
        mlflow.dspy.autolog()
        mlflow.set_experiment("Agent Assistant Bank Call")

        # Get prediction
        prediction = agent(speaker=speaker_id, utterance=utterance)
        
        # Store results in session state
        st.session_state.prediction_results = prediction
        st.session_state.analysis_complete = True
    else:
        st.warning("Please enter an utterance to analyze.")

# Display results if available
if st.session_state.analysis_complete and st.session_state.prediction_results:
    prediction = st.session_state.prediction_results
    
    st.subheader("Assistant Results")
    st.write(prediction.summary)
    st.markdown("---")
    
    # Add expandable sections for trajectory and reasoning
    with st.expander("💭 View Prediction Reasoning", expanded=False):
        st.write(prediction.reasoning)

    with st.expander("🔍 View Prediction Trajectory", expanded=False):
        st.write(prediction.trajectory)
    
    st.markdown("---")
    
    # MLflow button with callback
    if st.button("View MLflow Experiment Results", on_click=launch_mlflow):
        st.success("MLflow UI launched! Opening in new tab...")
else:
    if not utterance:
        st.warning("Please enter an utterance to analyze.")