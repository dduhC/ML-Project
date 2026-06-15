import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), max_df=0.9, min_df = 10)

    def preprocess(self, text):
        # Lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def fit_transform(self, texts):
        preprocessed_texts = [self.preprocess(text) for text in texts]
        return self.vectorizer.fit_transform(preprocessed_texts)

    def transform(self, texts):
        preprocessed_texts = [self.preprocess(text) for text in texts]
        return self.vectorizer.transform(preprocessed_texts)
    
preprocessor = TextPreprocessor()
