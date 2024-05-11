from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#Loading the pretrained SVM model and tf idf and count vectorizer to preprocess scraped paragraphs
tf_idf_loaded = joblib.load("extension/tfidf_transformer.joblib")
count_vectorizer_loaded = joblib.load("extension/count_vectorizer.joblib")
svm_model_loaded = joblib.load("extension/svm_model.joblib")

#The class from models.py is rewritten and changed to handle lists of str and to convert ndarray to list
class Preprocessing:
    def __init__(self):
        self.count_vectorizer = count_vectorizer_loaded
        self.tf_idf = tf_idf_loaded
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, text_data):
        return self.apply_methods(text_data)
    
    def vocab_size(self, text_data):
        all_words = " ".join(text_data).split()
        return len(set(all_words))

    def tokenization(self, text_data):
        corpus = nltk.word_tokenize(text_data)
        return corpus

    def lemmatization(self, text_data):
        corpus = [self.lemmatizer.lemmatize(word) for word in text_data]
        return corpus

    def delete_punctuation(self, text_data):
        if isinstance(text_data, str):
            corpus = text_data.lower()  
            corpus_without_punct = ''.join([' ' if char in string.punctuation else char for char in corpus])
            return corpus_without_punct
        elif isinstance(text_data, list):
        #If text_data is a list of strings
            processed_text = []
            for text in text_data:
                if isinstance(text, str):
                    corpus = text.lower()
                    corpus_without_punct = ''.join([' ' if char in string.punctuation else char for char in corpus])
                    processed_text.append(corpus_without_punct)
                else:
                # If any element in the list is not a string, return the original list
                    return text_data
            return processed_text
        else:
            return text_data  



    def delete_stopwords(self, text_data):
        if isinstance(text_data, list):
            cleaned_text = []
            for text in text_data:
                if isinstance(text, str):
                    stop_words = set(stopwords.words('english'))
                    cleaned_text.append(' '.join([word for word in text.split() if word not in stop_words]))
                else:
                    return text_data  
            return cleaned_text
        elif isinstance(text_data, str):
            stop_words = set(stopwords.words('english'))
            cleaned_text = ' '.join([word for word in text_data.split() if word not in stop_words])
            return cleaned_text
        else:
            return text_data  



    def apply_methods(self, text_data):
        text_data_deleted_punctuation = [self.delete_punctuation(data) for data in text_data]
        text_data_deleted_stopwords = [self.delete_stopwords(data) for data in text_data_deleted_punctuation]
        text_data_lemmatized = [' '.join(self.lemmatization(nltk.word_tokenize(data))) for data in text_data_deleted_stopwords]

    # Vectorization
        text_data_vector = self.count_vectorizer.transform(text_data_lemmatized).toarray()
        text_data_tfidf = self.tf_idf.transform(text_data_vector).toarray()

        return text_data_tfidf.tolist()  #Converting ndarray to list

preprocessor = Preprocessing()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scrape_and_classify', methods=['POST'])
def scrape_and_classify():
    if request.method != 'POST':
        return jsonify({'error': 'Method not allowed'}), 405

    try:
        data = request.get_json()
        url = data['url']

        #Scraping all paragraphs provided by background.js
        response = requests.get(url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch webpage.'}), 400

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]

        processed_paragraphs = []
        for paragraph in paragraphs:
            preprocessed_paragraph = preprocessor.apply_methods([paragraph])  #Preprocessing all paragraph
            prediction = svm_model_loaded.predict(preprocessed_paragraph)  # Getting the class for each paragraph using SVM model
            if prediction[0] == 'Addicted':
                continue 
            elif prediction[0] == 'Recovery':
                processed_paragraphs.append('<b>' + paragraph + '<b>')
            else: 
                processed_paragraphs.append(paragraph)

        # We need to return the processed paragraphs in json format to be able to be processed back by background.js
        return jsonify({'processed_paragraphs': processed_paragraphs}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
