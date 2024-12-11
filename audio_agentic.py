from bank_call_agent import AssistantAgent
import dspy
import mlflow
import subprocess
import time
import threading
from dotenv import load_dotenv
import os
import dspy
from dspy.utils.callback import BaseCallback
import json
from stt import recognize_from_file, transcription_manager
load_dotenv()

lm = dspy.LM(
    model=f"azure/{config.get('azure_deployment_model')}",
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_base=os.getenv('AZURE_OPENAI_API_BASE'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    cache=False
)
dspy.configure(lm=lm)
agent = AssistantAgent()
mlflow.dspy.autolog()
mlflow.set_experiment("Agent Assistant Bank Call - From Audio")

agent_threads = []

def transcriber_callback(transcription):
    # Handle different transcription types
    if transcription['type'] == 'interim':
    #    print(f"interim transcription: {transcription['text']}")
        pass
    
    elif transcription['type'] == 'final':
        # Concatenate final transcription and update display
        new_final = f"Speaker {transcription['speaker_id']}: {transcription['text']}\n"
        print(f"calling agent with {new_final}")
        def run_agent():
            prediction = agent(transcribed_text=new_final)
            print(f"got prediction: {prediction.relevant_information}")
        
        thread = threading.Thread(target=run_agent)
        thread.start()
        agent_threads.append(thread)

def main():
    audio_path = "audio.wav"
    # Configure LM
    lm = dspy.LM(
        model=f"azure/{config.get('azure_deployment_model')}",
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_base=os.getenv('AZURE_OPENAI_API_BASE'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        cache=False
    )
    dspy.configure(lm=lm)
    transcription_manager.set_consumer_callback(transcriber_callback)
    recognize_from_file(audio_path)
    for thread in agent_threads:
        thread.join()


if __name__ == "__main__":
    main()