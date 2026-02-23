from flask import Flask, render_template, request
import pickle
import sqlite3

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

# Create database
def init_db():
    conn = sqlite3.connect("news.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news TEXT,
            result TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    news = request.form['news']

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)[0]

    confidence = max(model.predict_proba(vector)[0]) * 100

    # Save to database
    conn = sqlite3.connect("news.db")
    c = conn.cursor()
    c.execute("INSERT INTO history (news, result) VALUES (?, ?)", (news, prediction))
    conn.commit()
    conn.close()

    return render_template(
        'result.html',
        news=news,
        result=prediction.upper(),
        confidence=round(confidence, 2)
    )

if __name__ == '__main__':
    app.run(debug=True)