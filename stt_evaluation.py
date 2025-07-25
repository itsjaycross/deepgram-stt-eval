"""
Multi-Vendor Speech-to-Text Evaluation Pipeline with Spanish Loanword Detection

This pipeline compares Deepgram and OpenAI Whisper STT models on a subset of the
MLCommons People's Speech dataset. It calculates standard metrics (WER, WRR, and SDI rates)
plus a custom Spanish loanword detection metric for technical evaluation.

Key Features:
- Simplified data loading from a local Hugging Face dataset.
- Core STT metrics: WER, WRR, Substitutions, Deletions, Insertions.
- Spanish loanword detection and accuracy metrics.
- Clean, readable, and streamlined code for technical submission.
- Updated for modern Deepgram (v3) and OpenAI Whisper SDKs.
- Text normalization applied to both reference and hypothesis transcripts.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import wave
from typing import Any, Dict, List, Tuple, Set

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from deepgram import DeepgramClient, PrerecordedOptions
from openai import OpenAI
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Evaluation Pipeline Class ---
class STTEvaluationPipeline:
    """
    A streamlined pipeline to evaluate and compare STT models.
    """

    SPANISH_LOANWORDS: Set[str] = {
        'adobe', 'aficionado', 'alamo', 'albino', 'alcove', 'alfalfa', 'alligator',
        'alpaca', 'amigo', 'amontillado', 'anchovy', 'armada', 'armadillo', 'arroyo',
        'avocado', 'banana', 'bandana', 'bandit', 'barbecue', 'barracuda', 'barrio',
        'bodega', 'bolero', 'bonanza', 'bonito', 'breeze', 'bronco', 'buckaroo',
        'burrito', 'burro', 'cabana', 'cabernet', 'cacao', 'cafeteria', 'caiman',
        'caldera', 'calico', 'canary', 'canasta', 'cannibal', 'canoe', 'canyon',
        'caramel', 'cargo', 'carnival', 'cassava', 'castanets', 'caudillo', 'cedar',
        'ceviche', 'chaparral', 'chaps', 'chayote', 'chihuahua', 'chile', 'chili',
        'chimichanga', 'chinchilla', 'chipotle', 'chocolate', 'churro', 'cigar',
        'cigarette', 'cilantro', 'cinch', 'cockroach', 'coco', 'cocoa', 'coconut',
        'cojones', 'colorado', 'condor', 'conquistador', 'cordoba', 'corral',
        'corsair', 'cortado', 'coyote', 'crimson', 'criollo', 'cumbia', 'daiquiri',
        'dengue', 'desperado', 'dorado', 'eldorado', 'embargo', 'empanada',
        'enchilada', 'escabeche', 'escapade', 'estancia', 'fajita', 'fandango',
        'fiesta', 'filibuster', 'flamingo', 'flan', 'flotilla', 'florida', 'galleon',
        'garbanzo', 'gazpacho', 'gringo', 'guacamole', 'guano', 'guava', 'guerrilla',
        'guitar', 'habanero', 'hacienda', 'hammock', 'hermano', 'hoosegow',
        'hurricane', 'iguana', 'incommunicado', 'indigo', 'jade', 'jaguar',
        'jalapeño', 'jerky', 'jicama', 'jojoba', 'junta', 'key', 'lariat', 'lasso',
        'latigo', 'lime', 'llama', 'llano', 'loco', 'machete', 'macho', 'madre',
        'maestro', 'maize', 'mamba', 'mambo', 'mango', 'manila', 'mano', 'mantilla',
        'mariachi', 'marijuana', 'marina', 'matador', 'maya', 'menudo', 'merengue',
        'mesa', 'mescal', 'mesquite', 'mestizo', 'mojito', 'mole', 'montana',
        'mosquito', 'mulatto', 'mustang', 'nacho', 'nada', 'negro', 'nevada',
        'nopal', 'ocelot', 'oregano', 'paella', 'palomino', 'pampa', 'papaya',
        'parakeet', 'parasol', 'patio', 'peccary', 'peso', 'peyote', 'picaresque',
        'picaro', 'pimento', 'pina', 'pinata', 'pinta', 'pinto', 'piranha', 'plantain',
        'platinum', 'plaza', 'politico', 'poncho', 'potato', 'presidio', 'pronto',
        'pueblo', 'puma', 'punctilio', 'python', 'quesadilla', 'quetzal', 'quinoa',
        'quixotic', 'ranch', 'ranchero', 'rapido', 'renegade', 'remuda', 'rico',
        'rodeo', 'rumba', 'sabino', 'salsa', 'samba', 'sangria', 'santa', 'sarsaparilla',
        'sassafras', 'savanna', 'savvy', 'serape', 'serrano', 'sherry', 'sierra',
        'siesta', 'silo', 'sombrero', 'sopapilla', 'stampede', 'stevedore', 'stockade',
        'suave', 'taco', 'tamale', 'tamarillo', 'tamarind', 'tango', 'tapas', 'tequila',
        'tobacco', 'tomatillo', 'tomato', 'toreador', 'tornado', 'torque', 'tortilla',
        'toucan', 'tuna', 'vamoose', 'vanilla', 'vaquero', 'vertigo', 'vicuna',
        'vigilante', 'villa', 'vista', 'yucca', 'zorro'
    }

    def __init__(self, deepgram_key: str, openai_key: str):
        """Initializes the evaluation pipeline with API keys and clients."""
        self.deepgram_client = DeepgramClient(deepgram_key)
        self.openai_client = OpenAI(api_key=openai_key)
        self.temp_dir = None
        logger.info("Deepgram and OpenAI clients initialized successfully.")

    def normalize_text(self, text: str) -> str:
        """
        Applies text normalization using NLTK for fair comparison between transcripts.
        This includes lowercasing, removing punctuation, expanding contractions,
        normalizing numbers, and standardizing whitespace.
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'ll": " will", "'ve": " have", "'re": " are", "'d": " would",
            "'m": " am", "it's": "it is", "let's": "let us",
            "that's": "that is", "what's": "what is", "there's": "there is",
            "here's": "here is", "where's": "where is", "who's": "who is",
            "how's": "how is", "y'all": "you all", "'cause": "because"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize numbers (convert digits to words for consistency)
        # This helps when one system outputs "5" and another outputs "five"
        digit_to_word = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
            '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '30': 'thirty',
            '40': 'forty', '50': 'fifty', '60': 'sixty', '70': 'seventy',
            '80': 'eighty', '90': 'ninety', '100': 'one hundred'
        }
        
        # Handle multi-digit numbers by converting to words
        words = text.split()
        normalized_words = []
        for word in words:
            if word.isdigit() and word in digit_to_word:
                normalized_words.append(digit_to_word[word])
            elif word.isdigit() and len(word) <= 2:
                # Handle two-digit numbers
                if int(word) <= 20:
                    normalized_words.append(word)  # Keep as is if not in map
                else:
                    tens = int(word[0]) * 10
                    ones = int(word[1])
                    if tens in digit_to_word and ones == 0:
                        normalized_words.append(digit_to_word[str(tens)])
                    elif tens in digit_to_word and ones in digit_to_word:
                        normalized_words.append(digit_to_word[str(tens)] + " " + digit_to_word[str(ones)])
                    else:
                        normalized_words.append(word)
            else:
                normalized_words.append(word)
        
        text = ' '.join(normalized_words)
        
        # Tokenize using NLTK
        tokens = word_tokenize(text)
        
        # Remove extra whitespace and rejoin
        text = ' '.join(tokens)
        
        # Final cleanup: remove multiple spaces and strip
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def load_test_data(self, max_samples: int = 50) -> pd.DataFrame:
        """Loads a small subset of the English Common Voice 11.0 test partition."""
        logger.info(f"Loading {max_samples} samples from English Common Voice 11.0 test partition...")
        try:
            dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="test", streaming=False, trust_remote_code=True)
            subset = dataset.select(range(min(len(dataset), max_samples)))
            
            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {self.temp_dir}")
            
            samples = []
            for i, item in enumerate(subset):
                try:
                    audio_path = os.path.join(self.temp_dir, f"audio_{i}.wav")
                    audio_array = item["audio"]["array"]
                    sampling_rate = item["audio"]["sampling_rate"]
                    
                    if audio_array.dtype != np.int16:
                        audio_array = (audio_array * 32767).astype(np.int16) if np.issubdtype(audio_array.dtype, np.floating) else audio_array.astype(np.int16)
                    
                    with wave.open(audio_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sampling_rate)
                        wav_file.writeframes(audio_array.tobytes())
                    
                    samples.append({
                        "audio_path": audio_path,
                        "transcript": item["sentence"].strip(),
                    })
                        
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue
            
            if samples:
                logger.info(f"Successfully loaded {len(samples)} samples.")
                return pd.DataFrame(samples)
            else:
                logger.error("No valid samples were loaded.")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}", exc_info=True)
            return pd.DataFrame()

    def cleanup_temp_files(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")

    async def transcribe_deepgram_async(self, audio_path: str) -> str:
        """Transcribes audio using the Deepgram API asynchronously."""
        try:
            with open(audio_path, 'rb') as audio_file:
                buffer_data = audio_file.read()

            payload = {"buffer": buffer_data}
            # UPDATED: Changed model from nova-2 to nova-3
            options = PrerecordedOptions(model="nova-3", smart_format=True, language="en")

            # CORRECTED: Use .asyncrest instead of the deprecated .asyncprerecorded
            response = await self.deepgram_client.listen.asyncrest.v("1").transcribe_file(
                payload, options
            )

            return response.results.channels[0].alternatives[0].transcript

        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            return ""

    async def transcribe_whisper(self, audio_path: str) -> str:
        """Transcribes audio using the OpenAI Whisper API."""
        try:
            with open(audio_path, "rb") as audio_file:
                # This is a synchronous call, but we'll await the async function
                response = self.openai_client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, response_format="text"
                )
            return response
        except Exception as e:
            logger.error(f"Whisper transcription failed for {audio_path}: {e}")
            return ""

    def detect_spanish_loanwords(self, text: str) -> List[str]:
        """Detects Spanish loanwords in the given text."""
        words = text.lower().split()
        found_loanwords = []
        for word in words:
            cleaned_word = word.strip('.,!?;:"\'')
            if cleaned_word in self.SPANISH_LOANWORDS:
                found_loanwords.append(cleaned_word)
        return found_loanwords

    def calculate_spanish_loanword_metrics(self, ref_loanwords: List[str], hyp_loanwords: List[str]) -> Dict[str, float]:
        """Calculates precision, recall, and F1 score for Spanish loanword detection."""
        ref_set = set(ref_loanwords)
        hyp_set = set(hyp_loanwords)
        
        if not ref_set and not hyp_set:
            return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'loanword_wer': 0.0}
        
        if not ref_set and hyp_set:
            return {'precision': 0.0, 'recall': 1.0, 'f1_score': 0.0, 'loanword_wer': 1.0}
        
        if ref_set and not hyp_set:
            return {'precision': 1.0, 'recall': 0.0, 'f1_score': 0.0, 'loanword_wer': 1.0}
        
        true_positives = len(ref_set.intersection(hyp_set))
        false_positives = len(hyp_set - ref_set)
        false_negatives = len(ref_set - hyp_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        loanword_errors = false_positives + false_negatives
        total_loanwords_in_ref = len(ref_set)
        loanword_wer = loanword_errors / total_loanwords_in_ref if total_loanwords_in_ref > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'loanword_wer': loanword_wer}

    def calculate_metrics(self, reference: str, hypothesis: str, normalize: bool = True) -> Dict[str, float]:
        """Calculates WER, WRR, SDI rates, and Spanish loanword metrics."""
        try:
            # Apply text normalization if requested
            if normalize:
                reference = self.normalize_text(reference)
                hypothesis = self.normalize_text(hypothesis)
            
            if not reference and not hypothesis:
                return {"wer": 0.0, "wrr": 1.0, "substitutions": 0.0, "deletions": 0.0, "insertions": 0.0, "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "loanword_wer": 0.0, 'spanish_loanwords_ref': 0, 'spanish_loanwords_hyp': 0}
            elif not reference:
                return {"wer": 1.0, "wrr": 0.0, "substitutions": 0.0, "deletions": 0.0, "insertions": 1.0, "precision": 0.0, "recall": 1.0, "f1_score": 0.0, "loanword_wer": 1.0, 'spanish_loanwords_ref': 0, 'spanish_loanwords_hyp': len(hypothesis.split())}
            elif not hypothesis:
                return {"wer": 1.0, "wrr": 0.0, "substitutions": 0.0, "deletions": 1.0, "insertions": 0.0, "precision": 1.0, "recall": 0.0, "f1_score": 0.0, "loanword_wer": 1.0, 'spanish_loanwords_ref': len(reference.split()), 'spanish_loanwords_hyp': 0}

            output = jiwer.process_words(reference, hypothesis)
            num_ref_words = len(reference.split()) if reference else 1

            substitutions_rate = output.substitutions / num_ref_words
            deletions_rate = output.deletions / num_ref_words
            insertions_rate = output.insertions / num_ref_words
            
            # CORRECTED: Manually calculate WRR from WER for compatibility
            wrr = 1.0 - output.wer

            ref_loanwords = self.detect_spanish_loanwords(reference)
            hyp_loanwords = self.detect_spanish_loanwords(hypothesis)
            loanword_metrics = self.calculate_spanish_loanword_metrics(ref_loanwords, hyp_loanwords)
            
            return {
                "wer": output.wer, 
                "wrr": wrr, # Use the calculated value
                "substitutions": substitutions_rate, 
                "deletions": deletions_rate, 
                "insertions": insertions_rate,
                **loanword_metrics,
                'spanish_loanwords_ref': len(ref_loanwords), 
                'spanish_loanwords_hyp': len(hyp_loanwords)
            }
        except Exception as e:
            logger.warning(f"Metric calculation failed: {e}. Returning defaults.")
            return {"wer": 1.0, "wrr": 0.0, "substitutions": 0.0, "deletions": 1.0, "insertions": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "loanword_wer": 1.0, 'spanish_loanwords_ref': 0, 'spanish_loanwords_hyp': 0}
    
    async def evaluate_models(self, max_samples: int = 50) -> pd.DataFrame:
        """Evaluates STT models on a dataset and calculates metrics."""
        logger.info(f"Starting STT model evaluation for {max_samples} samples...")
        data = self.load_test_data(max_samples)
        if data.empty:
            logger.error("No data loaded, aborting evaluation.")
            return pd.DataFrame()

        results = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing samples"):
            audio_path = row["audio_path"]
            reference = row["transcript"]
            try:
                # Await the async functions sequentially
                deepgram_transcript = await self.transcribe_deepgram_async(audio_path)
                whisper_transcript = await self.transcribe_whisper(audio_path)

                # Calculate metrics with normalization
                deepgram_metrics = self.calculate_metrics(reference, deepgram_transcript, normalize=True)
                whisper_metrics = self.calculate_metrics(reference, whisper_transcript, normalize=True)
                
                # Also store normalized versions for analysis
                normalized_reference = self.normalize_text(reference)
                normalized_deepgram = self.normalize_text(deepgram_transcript)
                normalized_whisper = self.normalize_text(whisper_transcript)
                
                results.append({
                    "audio_path": audio_path, 
                    "reference": reference,
                    "normalized_reference": normalized_reference,
                    "deepgram_transcript": deepgram_transcript, 
                    "normalized_deepgram": normalized_deepgram,
                    "whisper_transcript": whisper_transcript,
                    "normalized_whisper": normalized_whisper,
                    **{f"deepgram_{k}": v for k, v in deepgram_metrics.items()},
                    **{f"whisper_{k}": v for k, v in whisper_metrics.items()}
                })
            except Exception as e:
                logger.error(f"Error processing sample {audio_path}: {e}")
                continue
        
        return pd.DataFrame(results)

    def print_summary(self, results_df: pd.DataFrame):
        """Prints a clean, formatted summary of the evaluation results."""
        if results_df.empty:
            print("No results to summarize.")
            return

        summary = {
            "deepgram": {col.split('_', 1)[1]: results_df[col].mean() for col in results_df.columns if col.startswith('deepgram_') and pd.api.types.is_numeric_dtype(results_df[col])},
            "whisper": {col.split('_', 1)[1]: results_df[col].mean() for col in results_df.columns if col.startswith('whisper_') and pd.api.types.is_numeric_dtype(results_df[col])}
        }

        print("\n" + "="*60)
        print("                STT MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"\nEvaluated on {len(results_df)} audio samples.")
        print("Model: Deepgram Nova-3 vs OpenAI Whisper")
        print("Text normalization: ENABLED\n")
        
        print("STANDARD METRICS (with normalization):")
        print("-" * 50)
        print(f"{'Metric':<20} | {'Deepgram':<12} | {'Whisper':<12}")
        print("-" * 50)

        standard_metrics = ["wer", "wrr", "substitutions", "deletions", "insertions"]
        for metric in standard_metrics:
            dg_val = summary.get('deepgram', {}).get(metric, float('nan'))
            w_val = summary.get('whisper', {}).get(metric, float('nan'))
            print(f"{metric.upper():<20} | {dg_val:<12.3f} | {w_val:<12.3f}")

        print("\nSPANISH LOANWORD METRICS:")
        print("-" * 50)
        print(f"{'Metric':<20} | {'Deepgram':<12} | {'Whisper':<12}")
        print("-" * 50)
        
        loanword_metrics = ["precision", "recall", "f1_score", "loanword_wer"]
        for metric in loanword_metrics:
            dg_val = summary.get('deepgram', {}).get(metric, float('nan'))
            w_val = summary.get('whisper', {}).get(metric, float('nan'))
            display_name = metric.replace('_', ' ').title()
            print(f"{display_name:<20} | {dg_val:<12.3f} | {w_val:<12.3f}")

        print("-" * 50)
        print("\nLower is better for WER, S, D, I, and Loanword WER.")
        print("Higher is better for WRR, Precision, Recall, and F1.")
        
        if 'deepgram_spanish_loanwords_ref' in results_df.columns:
            samples_with_loanwords = results_df[results_df['deepgram_spanish_loanwords_ref'] > 0]
            print(f"\nNumber of samples containing Spanish loanwords: {len(samples_with_loanwords)}/{len(results_df)}")
        print("="*60)

    def visualize_metrics(self, results_df: pd.DataFrame):
        """Creates and saves a comprehensive visualization of all metrics."""
        if results_df.empty:
            logger.warning("Cannot create visualization from empty results.")
            return

        summary = {
            "deepgram": {col.split('_', 1)[1]: results_df[col].mean() for col in results_df.columns if col.startswith('deepgram_') and pd.api.types.is_numeric_dtype(results_df[col])},
            "whisper": {col.split('_', 1)[1]: results_df[col].mean() for col in results_df.columns if col.startswith('whisper_') and pd.api.types.is_numeric_dtype(results_df[col])}
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('STT Model Performance Comparison (Nova-3 vs Whisper)\nwith Text Normalization', fontsize=16, fontweight='bold')

        # Plot 1: WER and WRR
        wer_wrr_data = {'Metric': ['WER', 'WRR'] * 2, 'Value': [summary['deepgram']['wer'], summary['deepgram']['wrr'], summary['whisper']['wer'], summary['whisper']['wrr']], 'Model': ['Deepgram'] * 2 + ['Whisper'] * 2}
        wer_wrr_df = pd.DataFrame(wer_wrr_data)
        sns.barplot(data=wer_wrr_df, x='Metric', y='Value', hue='Model', ax=axes[0, 0])
        axes[0, 0].set_title('Overall Accuracy Metrics')
        axes[0, 0].set_ylim(0, 1.0)
        for container in axes[0, 0].containers:
            axes[0, 0].bar_label(container, fmt='%.3f')

        # Plot 2: SDI Rates
        sdi_data = {'Error Type': ['Substitutions', 'Deletions', 'Insertions'] * 2, 'Rate': [summary['deepgram']['substitutions'], summary['deepgram']['deletions'], summary['deepgram']['insertions'], summary['whisper']['substitutions'], summary['whisper']['deletions'], summary['whisper']['insertions']], 'Model': ['Deepgram'] * 3 + ['Whisper'] * 3}
        sdi_df = pd.DataFrame(sdi_data)
        sns.barplot(data=sdi_df, x='Error Type', y='Rate', hue='Model', ax=axes[0, 1])
        axes[0, 1].set_title('Error Breakdown (SDI Rates)')
        for container in axes[0, 1].containers:
            axes[0, 1].bar_label(container, fmt='%.3f')

        # Plot 3: Spanish Loanword Metrics
        loanword_data = {'Metric': ['Precision', 'Recall', 'F1 Score'] * 2, 'Value': [summary['deepgram']['precision'], summary['deepgram']['recall'], summary['deepgram']['f1_score'], summary['whisper']['precision'], summary['whisper']['recall'], summary['whisper']['f1_score']], 'Model': ['Deepgram'] * 3 + ['Whisper'] * 3}
        loanword_df = pd.DataFrame(loanword_data)
        sns.barplot(data=loanword_df, x='Metric', y='Value', hue='Model', ax=axes[1, 0])
        axes[1, 0].set_title('Spanish Loanword Recognition Performance')
        axes[1, 0].set_ylim(0, 1.0)
        for container in axes[1, 0].containers:
            axes[1, 0].bar_label(container, fmt='%.3f')

        # Plot 4: Loanword WER comparison
        loanword_wer_data = {'Model': ['Deepgram', 'Whisper'], 'Loanword WER': [summary['deepgram']['loanword_wer'], summary['whisper']['loanword_wer']]}
        loanword_wer_df = pd.DataFrame(loanword_wer_data)
        sns.barplot(data=loanword_wer_df, x='Model', y='Loanword WER', ax=axes[1, 1])
        axes[1, 1].set_title('Spanish Loanword Error Rate')
        axes[1, 1].set_ylim(0, max(loanword_wer_df['Loanword WER'].max() * 1.2, 0.1) if not loanword_wer_df['Loanword WER'].empty else 0.1)
        for container in axes[1, 1].containers:
            axes[1, 1].bar_label(container, fmt='%.3f')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('stt_evaluation_report.png', dpi=300)
        plt.close()
        logger.info("Visualization saved to stt_evaluation_report.png")

# --- Main Execution Block ---
if __name__ == "__main__":
    DEEPGRAM_API_KEY = "ENTER KEY HERE"
    OPENAI_API_KEY = "ENTER KEY HERE"

    if not DEEPGRAM_API_KEY or not OPENAI_API_KEY:
        logger.error("API keys are missing. Please set DEEPGRAM_API_KEY and OPENAI_API_KEY.")
    else:
        pipeline = None
        try:
            pipeline = STTEvaluationPipeline(deepgram_key=DEEPGRAM_API_KEY, openai_key=OPENAI_API_KEY)
            evaluation_results_df = asyncio.run(pipeline.evaluate_models(max_samples=250))

            if not evaluation_results_df.empty:
                evaluation_results_df.to_csv("stt_evaluation_results.csv", index=False)
                logger.info("Detailed results saved to stt_evaluation_results.csv")
                pipeline.print_summary(evaluation_results_df)
                pipeline.visualize_metrics(evaluation_results_df)
            else:
                logger.warning("Evaluation finished with no results.")
        except Exception as e:
            logger.error(f"The evaluation pipeline failed: {e}", exc_info=True)
        finally:
            if pipeline:
                pipeline.cleanup_temp_files()
