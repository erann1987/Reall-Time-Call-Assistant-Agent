# Bank Call Assistant

An AI-powered assistant that analyzes bank call conversations using DSPy and Azure OpenAI.

## Features

- Synthetic call data generation
- Speech-to-text conversion
- Real-time agent analysis of bank call utterances
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

4. Copy the `.env-example` file to `.env` and fill in the Azure OpenAI credentials:
```bash
cp .env-example .env
```

## Project Structure

- `synthetic_calls.jsonl`: Synthetic call data generated from a call transcript
- `prepare_vector_db.py`: Vector database preparation
- `app.py`: Main Streamlit application
- `stt.py`: Speech-to-text conversion
- `bank_call_agent.py`: DSPy agent implementation
- `config.yaml`: Configuration file for Azure OpenAI models usage
- `.env`: Configuration file for Azure OpenAI credentials
- `requirements.txt`: Project dependencies 

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```
