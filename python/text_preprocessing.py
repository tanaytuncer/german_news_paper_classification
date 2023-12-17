import nltk
import spacy
import pandas as pd

def drop_rows(df, text_col_name, label_names):  
    """
    Reorder loaded dataframe and drop all rows which includes null and duplicate values.
    Args:
        text_col_name (str): Column name which includes the news articles.
        label_names (ndarray): Column names which includes label names.
    Return:
        df (DataFrame): Dataframe
    """      
    df = df.melt(id_vars = text_col_name, value_name = "value", value_vars = label_names)
    df = df[df["value"] == 1]
    df = df.drop(labels = ["value"], axis = 1)
    df = df.dropna(subset = text_col_name)
    df = df.drop_duplicates()
    df.reset_index(inplace=True, drop=True)

    return df

def remove_stopwords(tokens):
    """
    Remove stopwords.
    Args:
        tokens (Pandas Series): List of tokens.
    Return:
        text (Pandas Series): List of tokens without stopwords
    """      
    spacy.prefer_gpu()
    ger_spacy = spacy.load("de_core_news_sm")
    ger_spacy_stopwords = ger_spacy.Defaults.stop_words
    text = [x for x in tokens if x not in ger_spacy_stopwords]
    return text

def lemmatize_words(tokens):
    """
    Lemmatize tokens to their respective lemma.
    Args:
        tokens (Pandas Series): List of tokens.
    Return:
        words (Pandas Series): List of lemmatized tokens.
    """      

    spacy.prefer_gpu()
    ger_spacy = spacy.load("de_core_news_sm")
    document = ger_spacy(" ".join(tokens)) 
    words = " ".join([token.lemma_ for token in document])
    return words

def extract_nouns(tokens):
    """
    Pos-Tag all tokens and extract only the nouns.
    Args:
        tokens (Pandas Series): List of tokens.
    Return:
        words (Pandas Series): List of tokens (nouns). 
    """      
    spacy.prefer_gpu()
    ger_spacy = spacy.load("de_core_news_sm", disable=["parser", "ner", "tokenizer", "textcat", "senter"]) 
        
    words = []
    for article in tokens:
        article = ger_spacy(article) 
        words.extend([token.text for token in article if token.pos_ == "NOUN"])

    return words
    
def split_data(X, y, emb = True, imbalanced = False):
    """
    Vectorize the input text data and split data into training, validation and test dataset.
    Args:
        X (Pandas Series): List of tokens.
        y (Pandas Series): Corresponding class name
        emb (boolean): If emb eq True then data is vectorized with CountVectorizer else TfidfVectorizer.
        imbalanced (boolean): If imbalanced eq True the RandomOverSampler is applied.
    Return:
        X, y, X_train, X_val, X_test, y_train, y_val, y_tes (array): X and y matrices for training, validation and test.
    """   
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import RandomOverSampler
    
    random_state = 2023

    if emb:
        count_vectorizer = CountVectorizer(max_features = 30000, ngram_range = (1,2), lowercase=False, binary = False)
        embedding = count_vectorizer.fit_transform(X)
    else:
        tf_idf_vectorizer = TfidfVectorizer(max_features = 30000, ngram_range = (1,2), lowercase=False)
        embedding = tf_idf_vectorizer.fit_transform(X)
                
    X = embedding.toarray()

    if imbalanced:
        o_smpl = RandomOverSampler(random_state = random_state) 
        X, y = o_smpl.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=random_state)
    
    print(f'Training instances: {X_train.shape[0]}, Validation instances: {X_val.shape[0]} and Test instances: {X_test.shape[0]}')

    return X, y, X_train, X_val, X_test, y_train, y_val, y_test

def principal_component_analysis(X, n_components):
    """
    Reduce the number of dimensions and extract the most important features with Principal Compoment Analysis.
    Args:
        X (array): Embedding or vectorized corpus 
        n_components (int): Number of dimensions 
    Return:
        X_transformed (array): Reduced dimensional input data.
    """   

    from sklearn.decomposition import PCA
    from tqdm import tqdm    
    
    for t in tqdm(range(100)):
        model = PCA(n_components = n_components, random_state = 2023)
        X_transformed = model.fit_transform(X)

    return X_transformed

def stemming_text(text):

    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("german")

    stemmed_text = "".join([stemmer.stem(word) for word in text])
    stemmed_text = [stemmer.stem(word) for word in stemmed_text]
    return stemmed_text

def remove_stopwords_punct(text):
    ger_spacy = spacy.load("de_core_news_sm")
    article = ger_spacy(text)
    text = [token.text for token in article if not token.is_stop and not token.is_punct]
    return text
