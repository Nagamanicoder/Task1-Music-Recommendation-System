from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os, json, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
app.secret_key = 'replace_this_with_a_strong_key'

BASE_DIR = os.path.dirname(__file__)
USERS_FILE = os.path.join(BASE_DIR, 'users.json')
SONGS_FILE = os.path.join(BASE_DIR, 'songs.csv')

# Load songs
songs = pd.read_csv(SONGS_FILE)
# create a 'combined' text column
text_cols = [c for c in songs.columns if songs[c].dtype == object]
songs['combined'] = songs[text_cols].fillna('').agg(' '.join, axis=1)

# TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(songs['combined'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE,'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE,'w') as f:
        json.dump(users,f, indent=2)

# Landing page
@app.route('/')
def landing():
    return render_template('landing.html')   # <- your landing page file

@app.route('/register', methods=['GET','POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        users = load_users()
        if any(u['username']==username for u in users):
            msg = 'Username already exists'
        else:
            users.append({'username': username, 'password': password})
            save_users(users)
            return redirect(url_for('login'))
    return render_template('register.html', msg=msg)

@app.route('/login', methods=['GET','POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        users = load_users()
        user = next((u for u in users if u['username']==username and u['password']==password), None)
        if user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            msg = 'Invalid credentials'
    return render_template('login.html', msg=msg)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    sample = songs.head(20).to_dict(orient='records')
    return render_template('dashboard.html', username=session['username'], songs=sample)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    title = data.get('title','')
    idx = None
    try:
        idx = songs[songs.apply(lambda row: title.lower() in ' '.join([str(row[c]) for c in songs.columns if row[c] is not None]).lower(), axis=1)].index[0]
    except Exception:
        return jsonify({'error':'song not found'}), 404
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_idx = [i for i,score in sim_scores[1:11]]
    recs = songs.iloc[top_idx][[c for c in songs.columns if c != 'combined']].to_dict(orient='records')
    return jsonify({'recs': recs})

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
