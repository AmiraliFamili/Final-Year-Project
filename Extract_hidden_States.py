import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional

class HiddenStateExtractor:
    """
    Extracts hidden states from specified transformer models
    Args:
        model_names: Dictionary with model names for QWEN and BERT
        device: Preferred computation device (default: 'cuda' if available)
    """
    def __init__(self, model_names: Dict[str, str], device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.tokenizers = {}
        
        # Load both models and tokenizers
        for model_type, model_name in model_names.items():
            self.tokenizers[model_type] = AutoTokenizer.from_pretrained(model_name)
            self.models[model_type] = AutoModel.from_pretrained(model_name).to(self.device)
            self.models[model_type].eval()  # Set to evaluation mode

    def _get_hidden_states(self, text: str, model_type: str) -> torch.Tensor:
        """
        Extract hidden states for a single text input
        Args:
            text: Input text string
            model_type: Type of model ('qwen' or 'bert')
        Returns:
            Tensor containing hidden states
        """
        tokenizer = self.tokenizers[model_type]
        model = self.models[model_type]
        
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Return last hidden states (adjust based on model architecture)
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    def process_text(self, text: str) -> Dict[str, List[float]]:
        """
        Process text through both models and return hidden states
        Args:
            text: Input text string
        Returns:
            Dictionary with hidden states from both models
        """
        features = {}
        
        # Get QWEN features
        qwen_hidden = self._get_hidden_states(text, 'qwen')
        features['qwen_hidden'] = qwen_hidden.cpu().numpy().flatten().tolist()
        
        # Get BERT features
        bert_hidden = self._get_hidden_states(text, 'bert')
        features['bert_hidden'] = bert_hidden.cpu().numpy().flatten().tolist()
        
        return features

class DatasetProcessor:
    """
    Processes dataset to add hidden state features
    Args:
        extractor: HiddenStateExtractor instance
        batch_size: Batch size for processing (default: 8)
    """
    def __init__(self, extractor: HiddenStateExtractor, batch_size: int = 8):
        self.extractor = extractor
        self.batch_size = batch_size

    def process_dataset(self, dataset: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Process entire dataset and add hidden state columns
        Args:
            dataset: Input DataFrame
            text_column: Name of column containing text
        Returns:
            DataFrame with added hidden state columns
        """
        processed_data = []
        
        # Process in batches
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset.iloc[i:i+self.batch_size]
            batch_texts = batch[text_column].tolist()
            
            batch_features = []
            for text in batch_texts:
                features = self.extractor.process_text(text)
                batch_features.append(features)
            
            # Add features to batch
            batch = batch.assign(
                qwen_hidden=[f['qwen_hidden'] for f in batch_features],
                bert_hidden=[f['bert_hidden'] for f in batch_features]
            )
            processed_data.append(batch)
        
        return pd.concat(processed_data, ignore_index=True)

if __name__ == "__main__":
    # Example usage
    model_names = {
        'qwen': 'Qwen/Qwen-7B',  # Replace with actual QWEN model name
        'bert': 'bert-base-uncased'
    }
    
    # Initialize extractor
    extractor = HiddenStateExtractor(model_names)
    
    # Load dataset
    dataset = pd.read_csv('your_dataset.csv')
    
    # Process dataset
    processor = DatasetProcessor(extractor)
    processed_df = processor.process_dataset(dataset, 'text_column_name')
    
    # Save processed data
    processed_df.to_parquet('dataset_with_hidden_states.parquet', index=False)