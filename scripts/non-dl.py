# ---------------------------------- Imports --------------------------------- #
import re
import numpy as np
import pandas as pd
import tqdm
from datasets import load_dataset

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier

# ------------------------------- Preprocessing ------------------------------ #
def ld_data():
    '''
    This function loads the "AiresPucrs/stanford-encyclopedia-philosophy" dataset 
    using the Hugging Face Datasets library, converts it into a pandas DataFrame, 
    and drops the 'metadata' column.

    Returns:
        pd.DataFrame: A DataFrame containing the dataset without the 'metadata' column.
    '''
    dataset = load_dataset("AiresPucrs/stanford-encyclopedia-philosophy", split = 'train')

    df = dataset.to_pandas()
    df.drop(columns=['metadata'], inplace=True)

    return df

# ------------------------- Lemmatizing and Stemming ------------------------- #
def lem_stem(df):
    '''
    Preprocess the text data in the DataFrame by tokenizing, removing stopwords, lemmatizing, stemming,
    and generating a final string representation of the lemmatized text.

    Args:
        df (pd.DataFrame): A DataFrame containing at least a 'text' column with raw text data.

    Returns:
        pd.DataFrame: The input DataFrame augmented with the following additional columns:
                      - 'tokenized_text': List of tokens after tokenization and stopword removal.
                      - 'lemmatized_text': List of tokens after lemmatization.
                      - 'stemmed_text': List of tokens after stemming.
                      - 'final_text': String obtained by joining the lemmatized tokens.
    '''
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    def remove_stopwords(tokens):
        return [token for token in tokens if token not in stop_words]
    
    df['tokenized_text'] = df['text'].apply(preprocess_tokenize)
    df['tokenized_text'] = df['tokenized_text'].apply(remove_stopwords)
    df['lemmatized_text'] = df['tokenized_text'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
    df['stemmed_text'] = df['tokenized_text'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])
    df['final_text'] = df['lemmatized_text'].apply(lambda tokens: ' '.join(tokens))

    return df

def preprocess_tokenize(text):
    '''
    Lowercase and tokenize input text after removing punctuation.

    Args:
        text (str): The input text string to process.

    Returns:
        list: A list of tokens (words) from the processed text.
    '''
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return tokens

# ---------------------------- Feature Engineering --------------------------- #
def feat_ext(df):
    '''
    Extract TF-IDF features from the preprocessed text data in the DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing at least:
                           - 'final_text': A string column with the processed text.
                           - 'category': The target labels for classification.

    Returns:
        X_tfidf (scipy.sparse matrix): The TF-IDF feature matrix.
        y (array-like): The target category labels.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer instance.
    '''
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),   # Range of Unigram to Trigram.
        max_features=15000,   # Limit to 15,000 terms.
        min_df=7,             # Ignore terms that appear in fewer than 7 documents.
        max_df=0.75,          # Ignore terms that appear in more than 75% of documents.
        use_idf=True,
        sublinear_tf=True
    )

    X_tfidf = vectorizer.fit_transform(df['final_text'])
    y = df['category']

    return X_tfidf, y, vectorizer

# ------------------------------- Data Splitting ----------------------------- #
def split_data(X_tfidf, y):
    '''
    Split the TF-IDF features and labels into training and testing sets.

    Args:
        X_tfidf (scipy.sparse matrix or array-like): The feature matrix (e.g., TF-IDF matrix).
        y (array-like): The target labels corresponding to the features.

    Returns:
        tuple: Four elements containing:
               - X_train: Training features.
               - X_test: Testing features.
               - y_train: Training labels.
               - y_test: Testing labels.
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

# ---------------------------- Class Weights ---------------------- #
def comp_class_weights(y_train):
    '''
    Compute sample weights and extract unique classes for balanced training.
    
    Args:
        y_train (array-like): The training labels.

    Returns:
        tuple: A tuple containing:
            - sample_weight (np.array): An array of sample weights corresponding to each element in y_train.
            - classes (np.array): An array of the unique classes found in y_train.
    '''
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    sample_weight = np.array([class_weight_dict[label] for label in y_train])

    return sample_weight, classes

# -------------------------- Training and Evaluation ------------------------- #
def train_eval_model(X_train, y_train, X_test, y_test, sample_weight, classes, vectorizer):
    '''
    Train and evaluate a text classification model using an SGDClassifier and output feature contributions for a specific category.

    Args:
        X_train (scipy.sparse matrix or array-like): Training feature matrix.
        y_train (array-like): Training labels.
        X_test (scipy.sparse matrix or array-like): Testing feature matrix.
        y_test (array-like): Testing labels.
        sample_weight (np.array): Array of sample weights corresponding to y_train.
        classes (np.array): Array of unique classes in the training labels.
        vectorizer (TfidfVectorizer): A fitted TF-IDF vectorizer used to extract feature names.

    Returns:
        tuple: A tuple containing:
            - clf (SGDClassifier): The trained classifier.
            - feature_names (np.array): An array of feature names from the vectorizer.
    '''
    clf = SGDClassifier(
        loss='hinge',     # Hinge loss.
        max_iter=1,       # One iteration per call to partial_fit.
        tol=None,         # Disable internal early stopping.
        warm_start=True,  # Retain model state between partial_fit calls.
        n_jobs=-1,
        random_state=42
    )

    n_epochs = 20
    for _ in tqdm.tqdm(range(n_epochs), desc="Training epochs"):
        clf.partial_fit(X_train, y_train, sample_weight=sample_weight, classes=classes)

    print("\nMaking predictions on test set...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    feature_names = np.array(vectorizer.get_feature_names_out())
    category_to_explain = 'abelard'
    if category_to_explain in clf.classes_:
        class_index = list(clf.classes_).index(category_to_explain)
        coefficients = clf.coef_[class_index]
        top_n = 10
        top_features_idx = np.argsort(coefficients)[-top_n:]
        top_features = feature_names[top_features_idx]
        
        print(f"\nTop features for category '{category_to_explain}':")
        for i, feature in enumerate(reversed(top_features)):
            coef = coefficients[top_features_idx[-(i+1)]]
            print(f"{feature}: {coef:.4f}")
    else:
        print(f"Category '{category_to_explain}' not found in the model's classes.")

    return clf, feature_names

# ---------------------------- Example Prediction ---------------------------- #
def predict_with_explanation(input_text, clf, vectorizer, feature_names):
    '''
    Predict the category of an input text and generate an explanation of the decision.

    Args:
        input_text (str): The raw input text to classify.
        clf (SGDClassifier): The trained classifier.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer used for transforming text.
        feature_names (np.array): Array of feature names corresponding to the TF-IDF vectorizer's vocabulary.

    Returns:
        tuple: A tuple containing:
            - predicted_category (str): The category predicted by the classifier.
            - explanation_str (str): A human-readable explanation indicating which features influenced the decision.
    '''
    X_input = vectorizer.transform([input_text])
    predicted_category = clf.predict(X_input)[0]
    category_index = list(clf.classes_).index(predicted_category)
    
    coefficients = clf.coef_[category_index]
    contributions = X_input.multiply(coefficients).toarray()[0]
    
    top_n_explain = 5
    top_indices = np.argsort(contributions)[-top_n_explain:]
    explanation_features = [(feature_names[i], contributions[i]) for i in top_indices if contributions[i] > 0]
    
    explanation_str = f"The input was classified as '{predicted_category}' because it contains indicative features such as: "
    explanation_str += ", ".join([f"{feat} (contribution: {coef:.4f})" for feat, coef in explanation_features])
    return predicted_category, explanation_str

# ----------------------------------- Main ----------------------------------- #
def main():
    df = ld_data()
    df = lem_stem(df)
    X_tfidf, y, vectorizer = feat_ext(df)
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y)
    sample_weight, classes = comp_class_weights(y_train)
    clf, feat_names = train_eval_model(X_train, y_train, X_test, y_test, sample_weight, classes, vectorizer)

    example_text = "In Earth's future, a global crop blight and second Dust Bowl are slowly rendering the planet uninhabitable."
    pred_cat, explanation = predict_with_explanation(example_text, clf, vectorizer, feat_names)
    print("\nPrediction and Explanation for example text:")
    print("Predicted Category:", pred_cat)
    print("Explanation:", explanation)

if __name__ == '__main__':
    main()
