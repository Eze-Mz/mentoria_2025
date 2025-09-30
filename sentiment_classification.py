#!/usr/bin/env python3
"""
Sentiment Classification Script for Obesity-related Comments
Using vLLM with quantized Llama model for offline inference
"""

import pandas as pd
import re
import logging
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObesitiySentimentClassifier:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", batch_size: int = 32):
        """
        Initialize the sentiment classifier
        
        Args:
            model_name: Name of the quantized Llama model to use
            batch_size: Number of comments to process in each batch
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.llm = None
        self.sampling_params = None
        self._setup_model()
        
    def _setup_model(self):
        """Setup the vLLM model and sampling parameters"""
        try:
            # Initialize vLLM with local model
            self.llm = LLM(
                model=self.model_name,
                # Remove quantization for local non-quantized model
                # quantization="awq",  # Enable if your model is quantized
                dtype="half",
                gpu_memory_utilization=0.8,
                max_model_len=2048,
                trust_remote_code=True  # For local models
            )
            
            # Configure sampling parameters for consistent output
            self.sampling_params = SamplingParams(
                temperature=0.1,  # Low temperature for consistent classification
                top_p=0.9,
                max_tokens=10,  # Short response needed
                stop=["<|eot_id|>", "\n", "###", "Clasificación:", "<|end_of_text|>"]
            )
            
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove link_url placeholders
        text = re.sub(r'\blink_url\b', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove URLs (additional cleanup)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation (keep some for context)
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\u00C0-\u017F\u0100-\u024F]', '', text)
        
        return text.strip()
    
    def is_valid_comment(self, text: str) -> bool:
        """
        Check if a comment is valid for sentiment analysis
        
        Args:
            text: Cleaned text to validate
            
        Returns:
            True if comment is valid, False otherwise
        """
        if not text or len(text.strip()) == 0:
            return False
        
        # Check if text has actual words (not just punctuation/numbers)
        words = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\b', text)
        if len(words) < 2:  # Require at least 2 words
            return False
        
        # Check minimum length
        if len(text.strip()) < 10:  # Require at least 10 characters
            return False
        
        # Check if it's mostly numbers or symbols
        alphanumeric_ratio = len(re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]', text)) / len(text)
        if alphanumeric_ratio < 0.3:  # Require at least 30% letters
            return False
        
        return True
    
    def create_prompt(self, comment: str) -> str:
        """
        Create a prompt for sentiment classification in Spanish
        
        Args:
            comment: The comment to classify
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Eres un experto en análisis de sentimientos. Tu tarea es clasificar el sentimiento de comentarios relacionados con obesidad y peso corporal.

Instrucciones:
- Clasifica el sentimiento como: POSITIVO, NEGATIVO, o NEUTRO
- POSITIVO: comentarios que expresan apoyo, comprensión, motivación o actitudes constructivas hacia personas con obesidad
- NEGATIVO: comentarios que expresan discriminación, burla, odio, desprecio o actitudes destructivas hacia personas con obesidad  
- NEUTRO: comentarios informativos, descriptivos o sin carga emocional clara

Responde únicamente con una de estas tres palabras: POSITIVO, NEGATIVO, NEUTRO<|eot_id|><|start_header_id|>user<|end_header_id|>

Comentario a clasificar: "{comment}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataset
        
        Args:
            df: DataFrame with comments
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Original dataset size: {len(df)}")
        
        # Clean the rawContent column
        df['cleaned_content'] = df['rawContent'].apply(self.clean_text)
        
        # Filter valid comments
        df['is_valid'] = df['cleaned_content'].apply(self.is_valid_comment)
        valid_df = df[df['is_valid']].copy()
        
        logger.info(f"Valid comments after preprocessing: {len(valid_df)}")
        logger.info(f"Removed {len(df) - len(valid_df)} invalid comments")
        
        return valid_df
    
    def classify_batch(self, comments: List[str]) -> List[str]:
        """
        Classify a batch of comments
        
        Args:
            comments: List of comments to classify
            
        Returns:
            List of sentiment classifications
        """
        prompts = [self.create_prompt(comment) for comment in comments]
        
        try:
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            classifications = []
            for output in outputs:
                response = output.outputs[0].text.strip().upper()
                
                # Extract classification from response
                if 'POSITIVO' in response:
                    classification = 'POSITIVO'
                elif 'NEGATIVO' in response:
                    classification = 'NEGATIVO'
                elif 'NEUTRO' in response:
                    classification = 'NEUTRO'
                else:
                    # Default to NEUTRO if unclear
                    classification = 'NEUTRO'
                    logger.warning(f"Unclear classification response: {response}")
                
                classifications.append(classification)
            
            return classifications
            
        except Exception as e:
            logger.error(f"Error in batch classification: {e}")
            # Return neutral for all in case of error
            return ['NEUTRO'] * len(comments)
    
    def classify_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify sentiment for entire dataset
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with sentiment classifications
        """
        logger.info("Starting sentiment classification...")
        
        comments = df['cleaned_content'].tolist()
        total_comments = len(comments)
        classifications = []
        
        # Process in batches
        for i in range(0, total_comments, self.batch_size):
            batch_end = min(i + self.batch_size, total_comments)
            batch_comments = comments[i:batch_end]
            
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_comments-1)//self.batch_size + 1}")
            
            batch_classifications = self.classify_batch(batch_comments)
            classifications.extend(batch_classifications)
        
        # Add classifications to DataFrame
        df['sentiment'] = classifications
        
        return df
    
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze classification results
        
        Args:
            df: DataFrame with sentiment classifications
            
        Returns:
            Dictionary with analysis results
        """
        sentiment_counts = df['sentiment'].value_counts()
        sentiment_percentages = df['sentiment'].value_counts(normalize=True) * 100
        
        analysis = {
            'total_comments': len(df),
            'sentiment_distribution': {
                'counts': sentiment_counts.to_dict(),
                'percentages': sentiment_percentages.round(2).to_dict()
            },
            'sample_comments': {}
        }
        
        # Get sample comments for each sentiment
        for sentiment in ['POSITIVO', 'NEGATIVO', 'NEUTRO']:
            if sentiment in df['sentiment'].values:
                sample = df[df['sentiment'] == sentiment]['cleaned_content'].head(3).tolist()
                analysis['sample_comments'][sentiment] = sample
        
        return analysis


def main():
    """Main execution function"""
    
    # Configuration
    DATA_PATH = "/home/ezequiel/ESTUDIO/trabajos_diplomatura/mentoria_2025/practicos/base_obesidad_completa.csv"
    OUTPUT_PATH = "/home/ezequiel/ESTUDIO/trabajos_diplomatura/mentoria_2025/practicos/base_obesidad_classified.csv"
    MODEL_NAME = "/home/ezequiel/SUSTANTIVA/sentiment_classification/huggingface-cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
    BATCH_SIZE = 50  # Adjust based on your GPU memory
    
    try:
        # Load data
        logger.info("Loading dataset...")
        df = pd.read_csv(DATA_PATH)
        
        # Initialize classifier
        logger.info("Initializing classifier...")
        classifier = ObesitiySentimentClassifier(model_name=MODEL_NAME, batch_size=BATCH_SIZE)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        processed_df = classifier.preprocess_data(df)
        
        # Classify sentiments
        logger.info("Classifying sentiments...")
        classified_df = classifier.classify_dataset(processed_df)
        
        # Save results
        logger.info(f"Saving results to {OUTPUT_PATH}...")
        classified_df.to_csv(OUTPUT_PATH, index=False)
        
        # Analyze results
        analysis = classifier.analyze_results(classified_df)
        
        # Print results
        print("\n" + "="*50)
        print("SENTIMENT CLASSIFICATION RESULTS")
        print("="*50)
        print(f"Total comments processed: {analysis['total_comments']}")
        print("\nSentiment Distribution:")
        for sentiment, count in analysis['sentiment_distribution']['counts'].items():
            percentage = analysis['sentiment_distribution']['percentages'][sentiment]
            print(f"  {sentiment}: {count} ({percentage}%)")
        
        print("\nSample Comments:")
        for sentiment, samples in analysis['sample_comments'].items():
            print(f"\n{sentiment}:")
            for i, sample in enumerate(samples, 1):
                print(f"  {i}. {sample[:100]}...")
        
        # Save analysis
        analysis_path = OUTPUT_PATH.replace('.csv', '_analysis.json')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Analysis saved to {analysis_path}")
        logger.info("Classification completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
