import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

labeled_df = pd.read_csv('tweets/labeled_tweets.csv')
labeled_df['Processed_Content'] = labeled_df['Content'].apply(preprocess_text)
labeled_df['Labels'] = labeled_df['Labels'].apply(lambda x: x.split(','))



print("Labeled Dataset:")
print(labeled_df.head())

mlb = MultiLabelBinarizer()
y_labeled = mlb.fit_transform(labeled_df['Labels'])

vectorizer = TfidfVectorizer()
X_labeled = vectorizer.fit_transform(labeled_df['Processed_Content'])

nb_clf = MultinomialNB()
multi_nb_clf = MultiOutputClassifier(nb_clf)
multi_nb_clf.fit(X_labeled, y_labeled)

print("Model trained on labeled data.")

unlabeled_df = pd.read_csv('tweets/2024-07-15_18-17-04_tweets_1-500.csv')
unlabeled_df['Processed_Content'] = unlabeled_df['Content'].apply(preprocess_text)

print("Unlabeled Dataset:")
print(unlabeled_df.head())

X_unlabeled = vectorizer.transform(unlabeled_df['Processed_Content'])

y_unlabeled_pred = multi_nb_clf.predict(X_unlabeled)
unlabeled_df['Predicted_Labels'] = mlb.inverse_transform(y_unlabeled_pred)

labeled_unlabeled_df = unlabeled_df[['Handle', 'Content']].copy()
labeled_unlabeled_df['Labels'] = unlabeled_df['Predicted_Labels'].apply(lambda labels: ','.join(labels))

print("Newly Labeled Dataset:")
print(labeled_unlabeled_df.head())

output_filepath = 'newly_labeled_tweets.csv'
labeled_unlabeled_df.to_csv(output_filepath, index=False, encoding='utf-8')

print(f"Newly labeled tweets saved to: {output_filepath}")

combined_df = pd.concat([labeled_df[['Handle', 'Content', 'Labels']], labeled_unlabeled_df], ignore_index=True)

X_combined = vectorizer.fit_transform(combined_df['Content'].apply(preprocess_text))

def find_similar_tweets(query, top_n=5):
    query_vec = vectorizer.transform([preprocess_text(query)])
    similarities = cosine_similarity(query_vec, X_combined).flatten()
    indices = similarities.argsort()[-top_n:][::-1]
    return combined_df.iloc[indices]

query = input("Enter your query: ")
similar_tweets = find_similar_tweets(query)
print("Similar Tweets:")
print(similar_tweets[['Content', 'Labels']])

similar_tweets_filepath = 'similar_tweets.csv'
similar_tweets.to_csv(similar_tweets_filepath, index=False, encoding='utf-8')
print(f"Similar tweets saved to: {similar_tweets_filepath}")

def rank_tweets(query):
    query_vec = vectorizer.transform([preprocess_text(query)])
    similarities = cosine_similarity(query_vec, X_combined).flatten()
    combined_df['Similarity'] = similarities
    return combined_df.sort_values(by='Similarity', ascending=False)

ranked_tweets = rank_tweets(query)
print("Ranked Tweets:")
print(ranked_tweets[['Content', 'Similarity']].head(10))

ranked_tweets_filepath = 'ranked_tweets.csv'
ranked_tweets.to_csv(ranked_tweets_filepath, index=False, encoding='utf-8')
print(f"Ranked tweets saved to: {ranked_tweets_filepath}")

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_combined)
combined_df['Cluster'] = kmeans.labels_

print("Clustered Tweets:")
print(combined_df[['Content', 'Cluster']].head(10))

clustered_tweets_filepath = 'clustered_tweets.csv'
combined_df.to_csv(clustered_tweets_filepath, index=False, encoding='utf-8')
print(f"Clustered tweets saved to: {clustered_tweets_filepath}")
