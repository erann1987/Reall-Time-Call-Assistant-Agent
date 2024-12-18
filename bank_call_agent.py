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
    """You are a ReACT (Reasoning and Action) agent designed to assist wealth management advisors during client calls by surfacing relevant information. Your primary goal is to enhance the advisor's efficiency and provide timely, accurate information. Here are your instructions:

    Real-Time Transcription Input:
    You will receive real-time transcribed text with the speaker id from the client-advisor call. This transcription will serve as the input for your reasoning and actions.
    
    Speaker Identification:
    You should reason about the speaker of the transcribed text. Only take into account utterances from the client.
    
    Intent Recognition:
    Carefully analyze the transcribed text to detect the client's intent and identify key topics being discussed.
    Do not proceed unless you have high confidence in understanding the client's specific intent.
    If the intent is ambiguous or unclear, continue listening and wait for more context.

    Note Retrieval:
    When you have clearly identified a specific client intent, you should retrieve relevant notes from the previous call:
    - Generating a focused query vector based on the confirmed intent and context
    - Searching the vector database for relevant notes from previous interactions
    If there is any uncertainty about the intent, wait for more transcribed text rather than performing premature searches.

    Tool Utilization:
    Use any of the provided tools to achieve the goal of surfacing relevant information. Select the most appropriate tool based on the identified intent and context.
    
    Display Relevant Information:
    Once relevant information is found based on a clear intent:
    - Dynamically display them to the advisor in real-time
    - Ensure they are clearly organized and directly related to the identified intent
    - Highlight the most relevant parts to help the advisor quickly grasp important information

    Citations:
    Include citations for the information you provide, indicating the sources of the generated information.
    
    Do not provide any recommendations or advice. Only provide information."""
    transcribed_text: str = dspy.InputField(desc="Recent transcribed text from the call")
    citations: str = dspy.OutputField(desc="The original observations used to generate the relevant information. If no relevant information is found, say 'None'")
    relevant_information: str = dspy.OutputField(desc="Concise and short summary of the relevant information. If no relevant information is found, say 'Waiting for more information'")



class AssistantAgent(dspy.Module):
    def __init__(self, results_from_search: int = 3, similarity_threshold: float = 1.0):
        self.results_from_search = results_from_search
        self.similarity_threshold = similarity_threshold
        self.agent = dspy.ReAct(
            signature=Assistant,
            tools=[self.retrieve_notes, self.stocks_info]
        )
    def forward(self, transcribed_text: str) -> str:
        return self.agent(transcribed_text=transcribed_text)
    
    def retrieve_notes(self, query: str) -> str | None:
        """Retrieve relevant notes from the previous call.
        Should always be used when you have a clear intent.

        Args:
            query (str): The query to search for in the notes.

        Returns:
            str: Relevant notes from the previous call with distance values."""
        search_results = retriever(query, k=self.results_from_search)
        search_results = [result for result in search_results if result['score'] <= self.similarity_threshold]
        if len(search_results) == 0:
            return None
        return  "\n\n".join([f"{result['long_text']}\nDistance: {result['score']}" for result in search_results])
    
    def stocks_info(self, stock_symbol: str) -> str:
        """Retrieve information about a stock.

        Args:
            stock_symbol (Literal['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'None specified']): The symbol of the stock to search for.

        Returns:
            str: Information about the stock."""
        
        stocks_info = {
            "AAPL": "Buy - Strong financial performance and growth potential.",
            "GOOGL": "Sell - Recent regulatory challenges and market competition.",
            "MSFT": "Hold - Stable performance with moderate growth prospects.",
            "AMZN": "Buy - Expanding market presence and innovative strategies.",
            "TSLA": "Sell - High volatility and uncertain future outlook.",
            "None specified": "Wait for specific stock information request."
        }
        return stocks_info.get(stock_symbol, "No information available for this stock.")

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
    cost = sum([token['cost'] for token in lm.history])
    mlflow.log_metric("cost", cost)
    print(lm.history)
    print(lm.history.keys())
    
