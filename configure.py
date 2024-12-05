from ollama import Client
import dspy
import subprocess

def pull_llama_model(model_name):
    try:
        # Run the ollama pull command
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True
        )

        # Check if the command was successful
        if result.returncode == 0:
            print(f"Successfully pulled model: {model_name}")
            print("Output:", result.stdout)
    except Exception as e:
        print(f"Error pulling model: {e}")
        raise

def configure_dspy_local_with_ollama(model):
    pull_llama_model(model)
    lm = dspy.LM(
        model=f'ollama_chat/{model}', 
        api_base='http://localhost:11434', 
        api_key=''
    )
    dspy.configure(lm=lm)

def configure_dspy_with_azure(deployment_name, azure_api_key, azure_api_base, azure_api_version):
    lm = dspy.LM(
        model=f"azure/{deployment_name}", 
        api_base=azure_api_base,
        api_version=azure_api_version,
        api_key=azure_api_key
    )
    dspy.configure(lm=lm)