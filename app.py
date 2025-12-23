# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    return render_template(
       'result.html',
       prediction=int(my_prediction[0])
    )


if __name__ == '__main__':
	app.run()
