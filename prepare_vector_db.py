from langchain_text_splitters import RecursiveCharacterTextSplitter
from dspy.retrieve.chromadb_rm import ChromadbRM
import chromadb.utils.embedding_functions as embedding_functions
import os
from dotenv import load_dotenv
import uuid
import chromadb
load_dotenv()

notes = [
    "Discussed conservative investment options, including fixed deposits and government bonds. Client interested in long-term, low-risk investments. Explained the benefits of each option, including guaranteed returns and minimal risk. Client will review the information and decide on the best option.",
    "Provided details on the conservative investment fund. Client considering starting with a minimum investment of $1,000. Explained the fund's asset mix, management team, and historical performance. Client appreciated the diversification and steady returns.",
    "Explained the benefits of high-quality corporate bonds. Client will review options and decide. Discussed the regular interest payments and low-risk nature of these bonds. Client found the steady income appealing.",
    "Discussed fixed deposit account with a 3.5% interest rate for five years. Client found it appealing. Explained the fixed interest rate and guaranteed returns. Client will consider opening a fixed deposit account.",
    "Provided information on government bonds and their low-risk nature. Client interested in steady returns. Discussed the tax benefits and stability of government bonds. Client will review the options and decide.",
    "Explained the process of opening a fixed deposit account. Client will visit the branch to proceed. Provided details on the required documentation and steps involved. Client appreciated the clear instructions.",
    "Discussed the conservative investment fund's asset mix and management. Client appreciated the diversification. Explained the fund's risk management strategies and historical performance. Client found the steady returns reassuring.",
    "Provided details on the minimum investment requirement for the conservative investment fund. Client considering it. Explained the benefits of starting with a small investment and gradually increasing it. Client found the flexibility appealing.",
    "Explained the benefits of diversifying investments across low-risk assets. Client found it beneficial. Discussed the importance of risk management and steady returns. Client will consider diversifying their investments.",
    "Discussed the stability and predictability of returns from fixed deposits. Client interested in guaranteed returns. Explained the fixed interest rate and minimal risk. Client will review the options and decide.",
    "Provided information on the interest rates for various fixed deposit terms. Client will decide on the term. Explained the benefits of different terms and their respective interest rates. Client appreciated the detailed information.",
    "Explained the risk assessment process for the conservative investment fund. Client willing to complete it. Provided details on the questionnaire and its purpose. Client found the process straightforward.",
    "Discussed the advantages of government bonds for conservative investors. Client found it suitable. Explained the stability and tax benefits of government bonds. Client will review the options and decide.",
    "Provided details on the conservative investment fund's performance history. Client impressed with the returns. Explained the fund's consistent performance and low-risk nature. Client found the historical data reassuring.",
    "Explained the process of applying for the conservative investment fund online. Client will consider it. Provided details on the required information and steps involved. Client appreciated the convenience of online application.",
    "Discussed the benefits of high-quality corporate bonds for steady income. Client interested in regular interest payments. Explained the low-risk nature and stability of these bonds. Client will review the options and decide.",
    "Provided information on the conservative investment fund's management team. Client appreciated the expertise. Explained the team's experience and track record. Client found the management team reassuring.",
    "Explained the tax benefits of investing in government bonds. Client found it advantageous. Discussed the stability and low-risk nature of government bonds. Client will review the options and decide.",
    "Discussed the conservative investment fund's risk management strategies. Client felt reassured. Explained the fund's approach to minimizing risk and ensuring steady returns. Client appreciated the detailed information.",
    "Provided details on the fixed deposit account's early withdrawal penalties. Client will consider the terms. Explained the penalties and their impact on returns. Client found the information helpful.",
    "Client reported unauthorized transactions on their account. Initiated dispute process. Verified the transactions and advised the client to monitor their account for further suspicious activity. Client appreciated the prompt assistance.",
    "Discussed mortgage loan options, including fixed-rate and adjustable-rate mortgages. Client will review. Provided details on interest rates, terms, and eligibility criteria. Client found the information helpful.",
    "Provided information on credit card rewards programs. Client interested in earning points. Explained the benefits and how to maximize rewards. Client will consider applying for a rewards credit card.",
    "Discussed personal loan options and interest rates. Client considering applying. Provided details on loan terms, eligibility, and repayment options. Client appreciated the clear information.",
    "Provided details on the bank's mobile banking app features. Client interested in using it. Explained the app's functionalities and benefits. Client will download and start using the app.",
    "Discussed the process of opening a new checking account. Client will visit the branch. Provided details on the required documentation and steps involved. Client appreciated the clear instructions.",
    "Provided information on the bank's savings account options. Client interested in higher interest rates. Explained the benefits of different savings accounts and their respective interest rates. Client will review the options and decide.",
    "Discussed the benefits of setting up automatic bill payments. Client found it convenient. Explained the process and how it can help avoid late fees. Client will consider setting up automatic payments.",
    "Provided details on the bank's online banking security features. Client appreciated the measures. Explained the security protocols and how to protect their account. Client found the information reassuring.",
    "Discussed the process of applying for a home equity loan. Client considering it. Provided details on loan terms, eligibility, and repayment options. Client appreciated the clear information.",
    "Provided information on the bank's travel insurance options. Client interested in coverage. Explained the benefits and how to apply for travel insurance. Client will review the options and decide.",
    "Discussed the benefits of the bank's retirement savings plans. Client considering starting one. Provided details on different plans and their benefits. Client found the information helpful.",
    "Provided details on the bank's student loan options. Client interested in financing education. Explained the loan terms, eligibility, and repayment options. Client appreciated the clear information.",
    "Discussed the process of transferring funds internationally. Client will proceed with the transfer. Provided details on the required information and steps involved. Client appreciated the clear instructions.",
    "Provided information on the bank's business loan options. Client considering expanding their business. Explained the loan terms, eligibility, and repayment options. Client found the information helpful.",
    "Discussed the benefits of the bank's wealth management services. Client interested in financial planning. Provided details on the services offered and their benefits. Client appreciated the comprehensive information.",
    "Provided details on the bank's auto loan options. Client considering purchasing a new car. Explained the loan terms, eligibility, and repayment options. Client found the information helpful.",
    "Discussed the process of setting up a joint account. Client will visit the branch with their partner. Provided details on the required documentation and steps involved. Client appreciated the clear instructions.",
    "Provided information on the bank's overdraft protection services. Client interested in avoiding fees. Explained the benefits and how to set up overdraft protection. Client found the information helpful.",
    "Discussed the benefits of the bank's health savings account. Client considering opening one. Provided details on the account features and benefits. Client appreciated the clear information."
]

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# chunks = text_splitter.create_documents([state_of_the_union])

documents = [
    {
        "details": note
    }
    for note in notes
]

# take from env
collection_name = os.getenv('DB_COLLECTION_NAME')
db_persist_path = os.getenv('DB_PERSIST_PATH')

ef = embedding_functions.OpenAIEmbeddingFunction (
    api_key=os.getenv('AZURE_API_KEY'),
    api_base=os.getenv('AZURE_API_BASE'),
    api_version=os.getenv('AZURE_API_VERSION'),
    api_type='azure',
    model_name=os.getenv('AZURE_EMBEDDING_MODEL')
)

chroma_client = chromadb.PersistentClient(path=db_persist_path)

try:
    collection = chroma_client.get_collection(collection_name, embedding_function=ef)
except:
    collection = chroma_client.create_collection(collection_name, embedding_function=ef)
    collection.add(
        documents=[d['details'] for d in documents],
        ids=[str(uuid.uuid4()) for _ in documents],
    )