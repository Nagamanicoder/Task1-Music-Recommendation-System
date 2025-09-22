
# Music Recommender Flask App

## What I created
- A starter Flask application with:
  - Register & Login pages (simple JSON store users.json)
  - Dashboard with search input to request recommendations
  - A content-based recommender using TF-IDF on all text columns in your `ex.csv`
  - Bootstrap-based responsive UI

## How to run
1. Create a Python virtual environment:
   python -m venv venv
   source venv/bin/activate   (Windows: venv\Scripts\activate)
2. Install dependencies:
   pip install flask scikit-learn pandas
3. Run:
   python app.py
4. Open http://127.0.0.1:5000

## Notes
- Passwords are stored in plaintext in users.json for simplicity; for production, use hashed passwords and a database.
- Recommendation searches for songs by title/artist substring and returns top similar tracks based on TF-IDF cosine similarity.
- I saved your uploaded dataset into `songs.csv` inside the project. Edit columns or adjust vectorization as needed.

