from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb.utils.embedding_functions as embedding_functions
import os
from dotenv import load_dotenv
import uuid
import chromadb
load_dotenv()
import yaml

# Load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

notes = [
    "The client expressed interest in conservative investment options with a focus on long-term stability and low risk. Discussed the fixed deposit account as a potential option, highlighting its guaranteed return and minimal risk.",
    "Provided detailed information about the fixed deposit account, including the term options and the fixed interest rate. Emphasized the higher return compared to a regular savings account",
    "Informed the client about the current interest rate for a five-year fixed deposit, which is 3.5% per annum. Highlighted the stability and predictability of the return.",
    "Expressed interest in NVIDIA stock and requested information about its performance and potential for growth.",
    "During the call, the client inquired about the current mortgage rates and the process of applying for a home loan. Provided information on fixed and variable mortgage rates and the required documentation for the application.",
    "Expressed interest in applying for a new credit card. Discussed the benefits of different credit card options, including cashback and rewards programs. Assisted the client in completing the application process.",
    "Requested information on setting up a repayment plan for an existing personal loan. Explained the available options for restructuring the loan and the impact on monthly payments and interest rates.",
    "Reported issues with accessing their online banking account. Provided troubleshooting steps and assisted in resetting the account password. Ensured the client could log in successfully.",
    "Asked about the benefits of opening a high-yield savings account. Discussed the interest rates, minimum balance requirements, and the process of transferring funds from a checking account.",
    "Requested a review of their current investment portfolio. Analyzed the performance of various assets and provided recommendations for rebalancing the portfolio to align with the client's financial goals.",
    "Inquired about the details of their existing insurance policy. Explained the coverage, premiums, and the process for filing a claim. Provided contact information for the insurance department.",
    "Expressed interest in setting up a business account for their new venture. Discussed the different types of business accounts available and the required documentation for opening an account.",
    "Asked about the bank's travel rewards program. Explained how to earn and redeem points for travel-related expenses, including flights and hotel stays. Provided information on the program's terms and conditions.",
    "Reported unauthorized transactions on their account. Initiated a fraud investigation and provided instructions on how to monitor the account for further suspicious activity. Advised the client to change their account passwords.",
    "Sought advice on retirement planning. Discussed different retirement savings options, including IRAs and 401(k) plans. Provided information on contribution limits and tax benefits.",
    "Inquired about the process of applying for an auto loan. Explained the loan terms, interest rates, and the required documentation. Assisted the client in completing the loan application.",
    "Asked about the process of exchanging foreign currency. Provided information on the current exchange rates and the fees associated with currency exchange services. Advised the client on the best time to exchange currency.",
    "Inquired about the features of the bank's mobile banking app. Explained how to use the app for various transactions, including bill payments and mobile check deposits. Provided troubleshooting tips for common issues.",
    "Expressed interest in refinancing their student loans. Discussed the benefits of refinancing, including lower interest rates and reduced monthly payments. Provided information on the application process.",
    "Inquired about the bank's wealth management services. Explained the different services available, including financial planning and investment management. Scheduled a follow-up meeting with a wealth advisor.",
    "Requested assistance with tax preparation. Provided information on the bank's tax preparation services and the required documentation. Scheduled an appointment with a tax advisor.",
    "Asked about the process of applying for a home equity line of credit (HELOC). Explained the terms, interest rates, and the required documentation. Assisted the client in completing the application.",
    "Inquired about the process of making charitable donations through their bank account. Provided information on the bank's donation services and the tax benefits of charitable giving.",
    "Requested information on closing their bank account. Explained the process and the required steps to ensure a smooth account closure. Provided information on transferring remaining funds to another account.",
]

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# chunks = text_splitter.create_documents([state_of_the_union])

documents = [
    {
        "details": note
    }
    for note in notes
]

ef = embedding_functions.OpenAIEmbeddingFunction (
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_base=os.getenv('AZURE_OPENAI_API_BASE'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    api_type='azure',
    model_name=config.get('azure_embedding_model')
)

chroma_client = chromadb.PersistentClient(path=config.get('db_persist_path'))

try:
    collection = chroma_client.get_collection(
        config.get('db_collection_name'), 
        embedding_function=ef
    )
except:
    collection = chroma_client.create_collection(
        config.get('db_collection_name'),
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef
    )
    collection.add(
        documents=[d['details'] for d in documents],
        ids=[str(uuid.uuid4()) for _ in documents],
    )