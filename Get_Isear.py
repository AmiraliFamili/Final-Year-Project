import pandas as pd
import re
import emoji
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class ISEARDataset:
    """
    A class to load, preprocess, and analyze the ISEAR (International Survey on Emotion Antecedents and Reactions) dataset.
    
    Attributes:
        df (pd.DataFrame): The processed DataFrame containing the dataset
        emotion_labels (Dict[int, str]): Mapping of emotion codes to labels
        coping_labels (Dict[int, str]): Mapping of coping mechanism codes to labels
        sex_labels (Dict[int, str]): Mapping of gender codes to labels
    """
    
    
        
    """
    ISEAR Constants
    """

    # --------------------------
    # Subject Demographics
    # --------------------------

    # COUN: Country codes and names
    COUNTRY_MAP = {
        1: "SWEDEN",
        2: "NORWAY",
        3: "F.R.G.",
        4: "FINLAND",
        5: "GREECE",
        6: "HONG KONG",
        7: "LEBANON",
        8: "AUSTRIA",
        9: "AUSTRALIA",
        10: "BRAZIL",
        11: "BOTSWANA",
        12: "BULGARIA",
        13: "FRANCE",
        14: "ITALY",
        15: "JAPAN",
        16: "NEW ZEALAND",
        17: "NETHERLANDS",
        18: "PORTUGAL",
        19: "SPAIN",
        20: "ZAMBIA",
        21: "ZIMBABWE",
        22: "USA",
        23: "POLAND",
        24: "NIGERIA",
        25: "ISRAEL",
        26: "INDIA",
        27: "MALAWI",
        28: "SWITZERLAND",
        29: "CHILE",
        30: "CHINA MAINLAND",
        31: "YUGOSLAVIA",
        32: "COSTA RICA",
        33: "HONDURAS",
        34: "MEXICO",
        35: "GUATEMALA",
        36: "VENEZUELA",
        37: "EL SALVADOR"
    }

    # SEX: Gender codes
    GENDER_MAP = {
        1: "MALE",
        2: "FEMALE"
    }

    # RELI: Religion codes
    RELIGION_MAP = {
        1: "PROTESTANT",
        2: "CATHOLIC",
        3: "JEWISH",
        4: "HINDU",
        5: "BUDDHIST",
        6: "NATIVE",
        7: "OTHERS",
        8: "ARELIGIOUS"
    }

    # PRAC: Practicing religion
    PRACTICE_MAP = {
        1: "TRUE",
        2: "FALSE"
    }

    # FOCC & MOCC: Parent occupation codes
    OCCUPATION_MAP = {
        1: "HOUSEWIFE",
        2: "UNEMPLOYED",
        3: "STUDENT",
        4: "BLUE COLLAR UNTRAINED",
        5: "BLUE COLLAR TRAINED",
        6: "WHITE COLLAR NONACADEMIC",
        7: "WHITE COLLAR ACADEMIC",
        8: "SELF-EMPLOYED NONACADEMIC",
        9: "SELF-EMPLOYED ACADEMIC"
    }

    # FIEL: Field of study
    FIELD_MAP = {
        1: "PSYCHOLOGY",
        2: "SOCIAL SCIENCES",
        3: "LANGUAGES",
        4: "FINE ARTS",
        5: "LAW",
        6: "NATURAL SCIENCE",
        7: "ENGINEERING",
        8: "MEDICAL",
        9: "OTHER"
    }

    # --------------------------
    # Emotion and Situation
    # --------------------------

    # EMOT: Emotion codes
    EMOTION_MAP = {
        1: "JOY",
        2: "FEAR",
        3: "ANGER",
        4: "SADNESS",
        5: "DISGUST",
        6: "SHAME",
        7: "GUILT"
    }

    # WHEN: When event occurred
    WHEN_MAP = {
        1: "DAYS AGO",
        2: "WEEKS AGO",
        3: "MONTHS AGO",
        4: "YEARS AGO"
    }

    # LONG: Duration of event
    DURATION_MAP = {
        1: "MINUTES",
        2: "HOUR",
        3: "HOURS",
        4: "DAY OR MORE"
    }

    # --------------------------
    # Reaction and Behavior
    # --------------------------

    # CON, EXPC, PLEA, FAIR, MORL, SELF: Control/Evaluation scales
    CONTROL_MAP = {
        0: "NA",
        1: "NONE",
        2: "A LITTLE",
        3: "VERY MUCH"
    }

    # RELA: Relationship impact
    RELATIONSHIP_MAP = {
        0: "NA",
        1: "NEGATIVE",
        2: "NONE",
        3: "POSITIVE"
    }

    # PLAN: Plan impact
    PLAN_MAP = {
        0: "NA",
        1: "HELPED",
        2: "NO MATTER",
        3: "HINDERED"
    }

    # CAUS: Cause of situation
    CAUSE_MAP = {
        0: "NA",
        1: "SELF",
        2: "CLOSE",
        3: "OTHER",
        4: "IMPERSONAL"
    }

    # COPING: Coping mechanism
    COPING_MAP = {
        1: "NO ACTION NECESSARY",
        2: "MANAGEABLE",
        3: "ESCAPABLE SITUATION",
        4: "DENIAL",
        5: "DOMINATED"
    }

    # --------------------------
    # Dataset Fields
    # --------------------------

    ISEAR_FIELDS = [
        "ID", "CITY", "COUN", "SUBJ", "SEX",
        "AGE", "RELI", "PRAC", "FOCC", "MOCC",
        "FIEL", "EMOT", "WHEN", "LONG", "INTS",
        "ERGO", "TROPHO", "TEMPER", "EXPRES",
        "MOVE", "EXP1", "EXP2", "EXP10", "PARAL",
        "CON", "EXPC", "PLEA", "PLAN", "FAIR",
        "CAUS", "COPING", "MORL", "SELF", "RELA",
        "VERBAL", "NEUTRO", "Field1", "Field3",
        "Field2", "MYKEY", "SIT", "STATE"
    ]

    # --------------------------
    # Value Ranges
    # --------------------------

    # Intensity ranges
    INTENSITY_RANGE = (1, 4)  # INTS
    ERGO_RANGE = (0, 4)       # ERGO
    TROPHO_RANGE = (0, 3)     # TROPHO
    TEMPER_RANGE = (-1, 2)    # TEMPER
    MOVE_RANGE = (-1, 1)      # MOVE
    EXPRES_RANGE = (0, 6)     # EXPRES
    PARAL_RANGE = (0, 3)      # PARAL

    # Binary fields
    BINARY_FIELDS = ["EXP1", "EXP2", "EXP10"]
        

    
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
    
    def __init__(self, filepath: str, delimiter: str = "|"):
        """
        Initialize the ISEAR dataset.
        
        Args:
            filepath: Path to the ISEAR dataset CSV file
            delimiter: Delimiter used in the CSV file (default is '|')
        """
        self.df = self._load_data(filepath, delimiter)
        self._preprocess_data()
        
    def _load_data(self, filepath: str, delimiter: str) -> pd.DataFrame:
        """Load and clean the raw dataset."""
        df = pd.read_csv(filepath, delimiter=delimiter)
        
        # Remove the very last unnecessary column if it exists
        if 'Unnamed: 42' in df.columns: 
            df = df.drop(columns=['Unnamed: 42'])
            
        # Drop duplicates and reset index
        df = df.drop_duplicates().reset_index(drop=True)
        
        return df
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        
        # Replace emoticons
        for emoticon, replacement in self.EMOTICON_MAPPING.items():
            text = text.replace(emoticon, replacement)
        
        # Handle emojis
        text = emoji.demojize(text)
        
        # Remove non-alphanumeric characters except spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _preprocess_data(self):
        """Apply all preprocessing steps to the dataset."""
        # Clean text
        self.df['clean_text'] = self.df['SIT'].apply(self._preprocess_text)
    
    def get_data(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        return self.df.copy()
    
# Example usage:
def get_isr():
    # Initialize the dataset
    isear = ISEARDataset("isear_dataset-master/isear.csv")
    
    # Get the processed data
    return isear.get_data()
