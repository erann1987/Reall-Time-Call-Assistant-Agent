import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import uuid
from dspy.retrieve.chromadb_rm import ChromadbRM
import dspy
import os
import mlflow
from dspy.utils.callback import BaseCallback
import json
from dotenv import load_dotenv
load_dotenv()
import yaml

# Load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

ef = embedding_functions.OpenAIEmbeddingFunction (
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_base=os.getenv('AZURE_OPENAI_API_BASE'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    api_type='azure',
    model_name=config.get('azure_embedding_model')
)

chroma_client = chromadb.PersistentClient(path=config.get('db_persist_path'))
retriever = ChromadbRM(
    collection_name=config.get('db_collection_name'),
    persist_directory=config.get('db_persist_path'),
    embedding_function=ef,
    client=chroma_client
)


class Assistant(dspy.Signature):
    """You are a ReACT (Reasoning and Action) agent designed to assist wealth management advisors during client calls by surfacing relevant notes from previous interactions. Your primary goal is to enhance the advisor's efficiency and provide timely, accurate information. Here are your instructions:

    Real-Time Transcription Input:
    You will receive real-time transcribed text with the speaker id from the client-advisor call. This transcription will serve as the input for your reasoning and actions.
  
    Intent Recognition:
    Carefully analyze the transcribed text to detect the client's intent and identify key topics being discussed.
    Do not proceed with searching unless you have high confidence in understanding the client's specific intent.
    If the intent is ambiguous or unclear, continue listening and wait for more context.

    Note Retrieval:
    Only when you have clearly identified a specific client intent, proceed with:
    - Generating a focused query vector based on the confirmed intent and context
    - Searching the vector database for relevant notes from previous interactions
    If there is any uncertainty about the intent, wait for more transcribed text rather than performing premature searches.
    
    Display Notes:
    Once relevant notes are found based on a clear intent:
    - Dynamically display them to the advisor in real-time
    - Ensure they are clearly organized and directly related to the identified intent
    - Highlight the most relevant parts to help the advisor quickly grasp important information
    Output language: English

    Citations:
    Include citations for the information you provide, indicating the sources of the retrieved notes and any other relevant data."""
    # speaker: str = dspy.InputField(desc="The speaker id of the utterance")
    transcribed_text: str = dspy.InputField(desc="Recent transcribed text from the call")
    citations: str = dspy.OutputField(desc="The original relevant notes that the relevant information is based on. If no relevant information is found, say 'None'")
    relevant_information: str = dspy.OutputField(desc="Concise and short summary of the relevant notes from previous calls. If no relevant information is found, say 'Waiting for more information'")



class AssistantAgent(dspy.Module):
    def __init__(self, similarity_threshold: float = 1.0):
        self.similarity_threshold = similarity_threshold
        self.agent = dspy.ReAct(
            signature=Assistant,
            tools=[self.retrieve_notes]
        )
    def forward(self, transcribed_text: str) -> str:
        return self.agent(transcribed_text=transcribed_text)
    
    def retrieve_notes(self, query: str) -> str:
        """Retrieve relevant notes from the previous call.

        Args:
            query (str): The query to search for in the notes.

        Returns:
            str: Relevant notes from the previous call with distance values."""
        search_results = retriever(query, k=config.get('db_n_results', 3))
        search_results = [result for result in search_results if result['score'] <= self.similarity_threshold]
        return  "\n\n".join([f"{result['long_text']}\nDistance: {result['score']}" for result in search_results])
    

# for testing
if __name__ == "__main__":
    class AgentLoggingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
        
        def on_module_end(self, call_id, outputs, exception):
            # Update the display with current step
            if "next_thought" in outputs:
                print(f"💭 Thinking: {outputs['next_thought']}")
            
            if "next_tool_name" in outputs:
                if outputs["next_tool_name"].lower() == "finish":
                    print("✅ Finish")
                else:
                    args_str = json.dumps(outputs.get("next_tool_args", {}), indent=2)
                    print(f"🔧 Using Tool: `{outputs['next_tool_name']}` with `{args_str}`")


    # Configure LM
    lm = dspy.LM(
        model=f"azure/{config.get('azure_deployment_model')}",
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_base=os.getenv('AZURE_OPENAI_API_BASE'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        cache=False
    )
    dspy.configure(lm=lm, callbacks=[AgentLoggingCallback()])

    mlflow.dspy.autolog()
    mlflow.set_experiment("Agent Assistant Bank Call")

    agent = AssistantAgent(similarity_threshold=1.0)

    utterance1 = "Bank Advisor: Good afternoon, thank you for calling ABC Bank. How can I assist you today?"
    prediction = agent(transcribed_text=utterance1)
    print(prediction)

    utterance2 = "Customer: Hi, I'm interested in conservative investments. Can you help?"
    prediction = agent(transcribed_text=utterance2)
    print(prediction)

    print(dspy.inspect_history(n=10))
