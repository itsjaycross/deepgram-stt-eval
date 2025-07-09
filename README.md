# Speech-to-Text Evaluation Pipeline

A Python-based pipeline for evaluating and comparing the performance of Deepgram and OpenAI Whisper speech-to-text (STT) models on English audio data.

## Features

- Evaluates STT models using the Common Voice 11.0 English test partition
- Calculates standard STT metrics (WER, WRR, SDI rates)
- Implements Spanish loanword detection and accuracy metrics
- Generates detailed visualizations of model performance
- Handles audio processing and transcription asynchronously
- Saves results for further analysis

## Requirements

- Python 3.13+
- Required packages (see [requirements.txt](requirements.txt)):
  - deepgram
  - openai
  - datasets
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - jiwer
  - tqdm
  - soundfile

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys:
   - Obtain a Deepgram API key from https://deepgram.com
   - Obtain an OpenAI API key from https://platform.openai.com
   - Update [stt_evaluation.py](stt_evaluation.py) with your API keys:
```python
DEEPGRAM_API_KEY = "your_deepgram_key_here"
OPENAI_API_KEY = "your_openai_key_here"
```

## Usage

Run the evaluation pipeline:
```bash
python stt_evaluation.py
```

The script will:
1. Load audio samples from Common Voice dataset
2. Transcribe audio using both Deepgram and Whisper
3. Calculate metrics including WER, WRR, and Spanish loanword detection
4. Generate visualizations comparing model performance
5. Save results to CSV and PNG files

## Output Files

- `stt_evaluation_results.csv`: Detailed transcription results and metrics
- `stt_evaluation_report.png`: Visualization of model performance comparison

## Metrics Calculated

- Word Error Rate (WER)
- Word Recognition Rate (WRR)
- Substitution, Deletion, and Insertion rates
- Spanish Loanword Detection:
  - Precision
  - Recall
  - F1 Score
  - Loanword-specific WER

## Visualization

The pipeline generates a comprehensive visualization showing:
1. Overall accuracy metrics (WER and WRR)
2. Error breakdown (SDI rates)
3. Spanish loanword recognition performance
4. Spanish loanword error rate comparison
