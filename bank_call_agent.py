import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import uuid
from dspy.retrieve.chromadb_rm import ChromadbRM
import dspy
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

documents =[
  {
    "date": "21.10.2024",
    "details": "Kunde ruft an und hat Interesse an konservativen Anlagemöglichkeiten. Erkläre dem Kunden unsere UBS Festgeldanlage. Kunde wird sich dies überlegen"
  },
  {
    "date": "15.08.2024",
    "details": "Kunde ruft an und möchte eine neue Debitkarte bestellen, da die jetzige defekt ist. Eine neue, kostenlose Karte für den Kunden bestellt"
  },
  {
    "date": "10.05.2024",
    "details": "Kunde hat das Digital Banking gesperrt und benötigt erneut einen Aktivierungscode. Aktivierungscode für den Kunden bestellt"
  }
]

collection_name = 'bank_call_agent'

ef = embedding_functions.OpenAIEmbeddingFunction (
    api_key=os.getenv('AZURE_API_KEY'),
    api_base=os.getenv('AZURE_API_BASE'),
    api_version=os.getenv('AZURE_API_VERSION'),
    api_type='azure',
    model_name=os.getenv('AZURE_EMBEDDING_MODEL')
)

db_persist_path = './bank_call_agent_db'
chroma_client = chromadb.PersistentClient(path=db_persist_path)

try:
    collection = chroma_client.get_collection(collection_name, embedding_function=ef)
except:
    collection = chroma_client.create_collection(collection_name, embedding_function=ef)
    collection.add(
        documents=[d['details'] for d in documents],
        ids=[str(uuid.uuid4()) for _ in documents]
    )


retriever = ChromadbRM(
    collection_name=collection_name,
    persist_directory=db_persist_path,
    embedding_function=ef,
    client=chroma_client
)

def retrieve_notes(query: str) -> str:
    """
    Retrieve relevant notes from the previous call.

    Args:
        query (str): The query to search for in the notes (Language: German).

    Returns:
        str: Relevant notes from the previous call with distance values.
    """
    search_results = retriever(query, k=3)
    return "\n\n".join([f"Note {i+1}: {d['long_text']}\nDistance: {d['score']}" for i, d in enumerate(search_results)])

def summarize_notes(relevant_notes: str) -> str:
    """
    Summarize relevant notes from the previous call.
    Provide a bullet point summary of the relevant notes retrieved from the call.
    Input:
        relevant_notes (str): Relevant notes from the previous call with low distance values.
    Output:
        str: A bullet points summary in English of the relevant notes retrieved from the call.
    """
    notes_summary_module = dspy.ChainOfThought(signature=NotesSummary)
    notes_summary = notes_summary_module(relevant_notes = relevant_notes)
    return notes_summary.summary

class NotesSummary(dspy.Signature):
    """
    Summarize relevant notes from the previous call.
    Provide a bullet point summary of the relevant notes retrieved from the call.
    """
    relevant_notes: str = dspy.InputField(desc="Relevant notes from the previous call with low distance values")
    summary: str = dspy.OutputField(desc="A bullet points summary in English of the relevant notes retrieved from the call")

class Assistant(dspy.Signature):
    """
    Based on recent utterances from a call between a client advisor and a client, think about what information might be needed from previous calls.
    If no information is needed, provide the following output: 'waiting for more information' and finish.
    If relevant notes were retrieved, summarize notes with low distance values, ignore notes with high distance values.
    """
    speaker: str = dspy.InputField(desc="The speaker id of the utterance")
    utterance: str = dspy.InputField(desc="Recent utterances from the call")
    summary: str = dspy.OutputField(desc="Relevant notes summary from the summarize_notes tool")


class AssistantAgent(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct(
            signature=Assistant,
            tools=[retrieve_notes, summarize_notes]
        )
    def forward(self, speaker: str, utterance: str) -> str:
        return self.agent(speaker=speaker, utterance=utterance)



# agent = AssistantAgent()

# utterance1 = "Hallo, ich möchte etwas Geld investieren, aber ich möchte etwas mit geringem Risiko. Können Sie mir mehr über die Möglichkeiten erzählen, die Sie haben, wie z.B. Kautionen oder ähnliches?"
# utterance2 = "Hallo Herr Jonson, guten Morgen, wie geht es Ihnen?"

# prediction = agent(speaker="2", utterance=utterance1)

# print(f"Prediction reasoning: {prediction.reasoning}")
# print(f"Prediction summary: {prediction.summary}")
# print(f"Prediction trajectory: {prediction.trajectory}")

# print(dspy.inspect_history(n=10))