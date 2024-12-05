# Bank Call Assistant

An AI-powered assistant that analyzes bank call conversations using DSPy and Azure OpenAI.

## Features

- Real-time analysis of bank call utterances
- Dynamic display of agent's thought process
- Detailed analysis results with trajectory and reasoning
- MLflow integration for experiment tracking

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd dspy-tutorial
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Azure OpenAI credentials:
```env
AZURE_API_BASE=your_api_base
AZURE_API_KEY=your_api_key
AZURE_API_VERSION=your_api_version
AZURE_DEPLOYMENT_MODEL=your_model_name
AZURE_EMBEDDING_MODEL=your_embedding_model
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Enter a speaker ID (1 for client, 2 for advisor) and the call utterance
3. Click "Analyze Utterance" to see the analysis
4. View the results and optionally explore the MLflow experiments

## Project Structure

- `app.py`: Main Streamlit application
- `bank_call_agent.py`: DSPy agent implementation
- `.env`: Configuration file for Azure OpenAI credentials
- `requirements.txt`: Project dependencies 