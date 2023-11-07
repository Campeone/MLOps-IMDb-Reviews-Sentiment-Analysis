import joblib 
import logging 
import tensorflow as tf
# from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from flask import Flask, render_template, request

app = Flask(__name__, static_folder='static')
# model and tokenizer file path
model_pth = 'models/IMDb_reviews_SANew.h5'
token_pth = 'models/vocabulary_tokens2.pkl'

# Load your sentiment analysis model
model = tf.keras.models.load_model(model_pth)

# Load your vocabulary tokens using joblib or any other method you prefer
vocabulary_tokens = joblib.load(token_pth) 

logging.basicConfig(filename="imdb.log", level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s') 



@app.route('/')
def home(): 
    # log an info message
    app.logger.info('User accessed the homepage.') 
    # return the home page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict(): 
    if request.method == 'POST': 

        # Get the movie name and review from the front-end
        movie_name = request.form.get('movieName')
        movie_review = request.form.get('review')
           
        # Tokenize and preprocess the review text
        review_sequence = vocabulary_tokens.texts_to_sequences([movie_review])
        review_padded = pad_sequences(review_sequence, maxlen=500)

        # log the movie name and review
        app.logger.info(f'Sentiment Analysis requested for this review: "{movie_review}" for a movie titled {movie_name}')
        
        # Perform sentiment analysis using your model
        pred_prob = model.predict(review_padded)

        # Determine sentiment based on probabilities 
        deg = []
        if pred_prob[0][1] > 0.5: 
            sentimentR = "Positive" 
            deg.append(pred_prob[0][1]) 
        else: 
            sentimentR = "Negative" 
            deg.append(pred_prob[0][0]) 
        
        output = f'The predicted sentiment is {sentimentR}, with a confidence level of {round(float(deg[0])*100, 2)}%'

        # log the prediction probabilities and the predicted sentiment
        app.logger.info(f'And got a {pred_prob} prediction probabilities, and a {sentimentR} sentiment')

        return render_template('index.html', sentiment = output) 
    
     # For the initial GET request (before the form is submitted), provide a default value
    return render_template('index.html', sentiment="yeA")
if __name__ == "__main__":
    app.run(debug=True)
