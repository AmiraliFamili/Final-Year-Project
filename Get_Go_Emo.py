import re
import pandas as pd 
import emoji
from typing import List, Dict, Union

class GoEmotionDataset:
    """
    A class to load, preprocess, and analyze the GoEmotions dataset.
    
    Attributes:
        emotions (List[str]): List of emotion labels
        positive_emotions (set): Set of positive emotions
        ambiguous_emotions (set): Set of ambiguous emotions
        negative_emotions (set): Set of negative emotions
        df (pd.DataFrame): The processed DataFrame containing the dataset
    """
    
    EMOTIONS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    POSITIVE_EMOTIONS = {
        "admiration", "amusement", "approval", "caring", "desire", "excitement",
        "gratitude", "joy", "love", "optimism", "pride", "relief"
    }
    
    AMBIGUOUS_EMOTIONS = {
        "confusion", "curiosity", "surprise", "realization", "neutral"
    }
    
    NEGATIVE_EMOTIONS = {
        "anger", "annoyance", "disappointment", "disapproval", "disgust",
        "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"
    }
    
    EMOTICON_MAPPING = {
        ":)": "happy", ":))": "happy", ":-)": "happy", ":-))": "happy",
        ":(": "sad", ":((": "sad", ":-((": "sad", ":-((": "sad",
        ":/": "confusion", "://": "confusion", ":-/": "confusion",
        ":-\'": "confusion", ":-//": "confusion", ":\\": "confusion",
        ":-\\": "confusion", ":|": "neutral", ":-|": "neutral",
        "XD": "laugh", ":D": "laugh", ":-D": "laugh",
        ">": "more than", "<": "less than", "<=": "less than or equal",
        ">=": "more than or equal", "=": "equal", "==": "is equivalent to"
    }
    
    def __init__(self, train_path: str, test_path: str, val_path: str):
        """
        Initialize the dataset by loading and processing the data.
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
            val_path: Path to validation data CSV
        """
        self.df = self._load_data(train_path, test_path, val_path)
        self._preprocess_data()
        
    def _load_data(self, train_path: str, test_path: str, val_path: str) -> pd.DataFrame:
        """Load and combine the dataset components."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        val_df = pd.read_csv(val_path)
        
        main_df = pd.concat([train_df, test_df, val_df], axis=0)
        main_df = main_df.reset_index(drop=True)
        main_df.drop_duplicates(inplace=True)
        
        return main_df
    
    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        
        # Replace emoticons
        for emoticon, replacement in GoEmotionDataset.EMOTICON_MAPPING.items():
            text = text.replace(emoticon, replacement)
        
        # Handle emojis
        text = emoji.demojize(text)
        
        # Remove unwanted punctuations (keeping basic sentence punctuations)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def _string_to_list(label_str: str) -> List[int]:
        """Convert string representation of array to list of integers."""
        return [int(x) for x in label_str.strip('[]').replace(',', '').split()]
    
    def _preprocess_data(self):
        """Apply all preprocessing steps to the dataset."""
        # Clean text
        self.df['clean_text'] = self.df['text'].apply(self._preprocess_text)
        
        # Process labels and emotions: convert the labels to proper lists.
        # self.df['labels'] = self.df['labels'].apply(self._string_to_list)
        
        self.df = self.df.drop(columns=['text', 'id'])
    
    def _get_sentiment_category(self, emotion_list: List[str]) -> str:
        """Categorize the overall sentiment of a text based on its emotions."""
        if any(emotion in self.POSITIVE_EMOTIONS for emotion in emotion_list):
            return "positive"
        elif any(emotion in self.NEGATIVE_EMOTIONS for emotion in emotion_list):
            return "negative"
        elif any(emotion in self.AMBIGUOUS_EMOTIONS for emotion in emotion_list):
            return "ambiguous"
        return "neutral"
    
    def get_data(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        return self.df.copy()


# Example usage:
def get_go(device_path=''):
    dataset = GoEmotionDataset(
        train_path=device_path+'Go_Emotion_Google/go_emotions_train.csv',
        test_path=device_path+'Go_Emotion_Google/go_emotions_test.csv',
        val_path=device_path+'Go_Emotion_Google/go_emotions_validation.csv'
    )
    
    return dataset.get_data()