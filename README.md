# Speech-to-Text (STT) Vendor Evaluation Pipeline

A Python-based pipeline for benchmarking the performance of Deepgram (Nova-2) and OpenAI (Whisper-1) speech-to-text models on English audio data.

This project provides a framework for evaluating STT vendors on both standard accuracy metrics and custom linguistic challenges. The goal is to produce clear, data-driven insights to inform model selection for production use cases.

## Features

- Evaluates STT models using a subset of the Common Voice 11.0 English test partition.
- Applies NLTK-based text normalization to ensure fair comparison between models.
- Calculates standard STT metrics (WER, WRR, and SDI rates) using the jiwer library.
- Implements a custom metric for Spanish loanword detection to test performance on specialized vocabularies.
- Generates a detailed CSV report with all transcripts and metrics for granular analysis.
- Creates a multi-panel PNG visualization comparing model performance.
- Uses asyncio to run API calls concurrently for improved efficiency.

## Requirements

- Python 3.10+
- All required packages are listed in [requirements.txt](requirements.txt)

## Setup

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Set up API Keys

This project requires API keys from Deepgram and OpenAI. It is configured to load these keys from environment variables for security.

#### Recommended Method: Create a `.env` file

Create a `.env` file in the root of the project directory:

```env
# .env
DEEPGRAM_API_KEY="your-deepgram-key-here"
OPENAI_API_KEY="your-openai-key-here"
```

The script will automatically load these keys if python-dotenv is installed.

#### Alternative Method: Set environment variables directly

Set the environment variables directly in your terminal session:

```bash
export DEEPGRAM_API_KEY="your-deepgram-key-here"
export OPENAI_API_KEY="your-openai-key-here"
```

## Usage

Run the evaluation pipeline from your terminal:

```bash
python stt_evaluation.py
```

The script will execute the following steps:

1. Load and prepare audio samples from the Common Voice dataset.
2. Transcribe each sample using both the Deepgram and Whisper APIs.
3. Calculate a suite of performance metrics for each transcription.
4. Generate and save a summary visualization (`stt_evaluation_report.png`).
5. Save the detailed, sample-by-sample results to a CSV file (`stt_evaluation_results.csv`).
6. Print a summary of the findings to the console.

## Output Files

- `stt_evaluation_results.csv`: A detailed CSV file containing the reference transcript, vendor hypotheses, and all calculated metrics for every sample.
- `stt_evaluation_report.png`: A multi-panel plot visualizing the comparative performance of the models.

## Metrics Calculated

### Standard Metrics

- **Word Error Rate (WER)**: The primary indicator of overall accuracy.
- **Word Recognition Rate (WRR)**: The percentage of words correctly transcribed (1 - WER).
- **Substitution Rate**: Rate of incorrect word substitutions.
- **Deletion Rate**: Rate of missed words.
- **Insertion Rate**: Rate of extra words inserted.

### Custom Metric: Spanish Loanword Performance

- **Precision**: Of the words identified as loanwords, how many were correct?
- **Recall**: Of the actual loanwords present, how many were found?
- **F1-Score**: The balanced harmonic mean of Precision and Recall.
- **Loanword WER**: A focused WER calculated only on the loanword vocabulary.
