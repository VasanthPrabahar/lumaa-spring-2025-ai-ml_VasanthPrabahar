# Content-Based Movie Recommendation System

## Overview
This project is a content-based movie recommendation system that takes a user's input description of preferred movie genres, keywords, or settings and returns the top 5 recommended movies based on textual similarity.

## Dataset
The dataset used is `Movies List.csv`, which includes movie information such as genres, keywords, overviews, and cast.

## Installation
1. Install the required packages by running:
```bash
pip install pandas numpy nltk scikit-learn
```
2. Download NLTK stopwords and punkt tokenizer:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Code Explanation

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
```
- **Pandas** for data manipulation
- **Numpy** for numerical operations
- **NLTK** for text preprocessing
- **Scikit-learn** for TF-IDF vectorization and cosine similarity

### 2. Load the Dataset
```python
df = pd.read_csv('/content/Movies List.csv')
```
This line loads the dataset into a DataFrame.

### 3. Text Preprocessing
```python
def clean_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text
```
- Converts text to lowercase
- Removes punctuation
- Removes stopwords (common words like 'the', 'and', etc.)

### 4. Combine Relevant Columns
```python
df['combined'] = df['genres'].fillna('') + ' ' + df['keywords'].fillna('') + ' ' + df['overview'].fillna('') + ' ' + df['cast'].fillna('')
df['combined'] = df['combined'].apply(clean_text)
```
- Merges genres, keywords, overview, and cast into a single text field
- Applies text cleaning

### 5. Text Vectorization (TF-IDF)
```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['combined'])
```
- Converts text into numerical vectors using TF-IDF

### 6. Movie Recommendation Function
```python
def recommend_movies():
    query = input("Enter a description of the type of movies you like: ")
    query = clean_text(query)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    top_indices = similarities.argsort()[-5:][::-1]
    recommendations = df.iloc[top_indices][['title']].to_dict('records')
    print("Top movie recommendations:")
    for i, movie in enumerate(recommendations, start=1):
        print(f"{i}. {movie['title']}")
```
- Takes a user input query
- Cleans the query
- Calculates similarity scores with the dataset
- Returns the top 5 recommended movie titles

### 7. Display Salary Expectation
```python
print('Salary expectation per month : 4000 USD')
```

## Running the Code
To run the code, simply execute:
```bash
python recommend_movies.py
```
Follow the prompt to enter your movie preferences.

## Example Input and Output
**Input:**
```
Enter a description of the type of movies you like: I love thrilling horror movies
```
**Output:**
```
Top movie recommendations:
1. Grindhouse
2. Blow Out
3. Insidious: Chapter 3
4. Love and Death on Long Island
5. Transsiberian
```

## Author
Vasanth Prabahar

## License
This project is open-source and available for use and modification.
