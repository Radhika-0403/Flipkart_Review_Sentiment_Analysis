from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load your trained model
model = joblib.load("E:/Innomatics Research Labs/Sentiment_Analysis/ML_Inference/Model/Model.pkl")

# Function for preprocessing text data
def clean(doc):
    # This text contains a lot of <br/> tags.
    doc = doc.replace("</br>", " ")
    
    # Remove punctuation and numbers.
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])

    # Converting to lower case
    doc = doc.lower()
    
    # Tokenization
    tokens = nltk.word_tokenize(doc)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
    
    # Join and return
    return " ".join(filtered_tokens)
def map_label_to_sentiment(label):
    if label > 3:
        return 'Positive \U0001F604'
    elif label == 3:
        return 'Neutral \U0001F610'
    else:
        return 'Negative \U0001F641' 

@app.route('/', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        text = request.form['text']
        
        preprocessed_text = clean(text)
        
        
        prediction = model.predict([preprocessed_text])
        
        
        result = map_label_to_sentiment(prediction[0])
        


    return render_template('result.html', text=text, result=result)

@app.route('/result')
def result():
    
    return render_template('result.html')

@app.route('/clean', methods=['POST'])
def clean_data():
    
    new_data = request.form.getlist('new_data')
    
    
    new_data_clean = [clean(doc) for doc in new_data]
    
    
    predictions = model.predict(new_data_clean)
    
    
    results = [map_label_to_sentiment(pred) for pred in predictions]
    
    
    print("Predictions:", results)
    
    
    return jsonify(predictions=results)

if __name__ == '__main__':
    app.run(debug=True)
