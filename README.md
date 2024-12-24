# Bank Call Assistant

An AI-powered assistant that analyzes bank call conversations in real-time using DSPy and Azure OpenAI.

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

5. Copy the `config-example.yaml` file to `config.yaml` and fill in the Azure OpenAI models and vector database details:
    ```bash
    cp config.yaml.example config.yaml
    ```

6. Run the `synthetic_call_transcription.py` script to generate synthetic call audio files and motes:
    ```bash
    python synthetic_call_transcription.py --conversation-topic "Conservative Investing"
    ```

7. Run the `prepare_vector_db.py` script to prepare the vector database:
    ```bash
    python prepare_vector_db.py --notes-file synthetic_data/Conservative Investing/call_notes.txt
    ```

## Project Structure

- `synthetic_call_transcription.py`: Synthetic call data generation and transcription
- `prepare_vector_db.py`: Vector database preparation
- `app.py`: Main Streamlit application
- `stt.py`: Speech-to-text conversion
- `bank_call_agent.py`: DSPy agent implementation
- `config.yaml`: Configuration file for Azure OpenAI models usage
- `.env`: Configuration file for Azure OpenAI credentials
- `requirements.txt`: Project dependencies 

## Usage

1. Run the vector database preparation script:
    ```bash
    python prepare_vector_db.py
    ```

2. [OPTIONAL] Run the synthetic call generation script to generate synthetic call audio file:
    ```bash
    python synthetic_call_generation.py
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
