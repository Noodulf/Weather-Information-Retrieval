import pandas as pd
import os

directory = r'C:\Users\manas\Desktop\project\selenium-twitter-scraper\tweets'
filename = '2024-07-15_10-27-40_tweets_1-1291.csv'
filepath = os.path.join(directory, filename)

print(f"File path: {filepath}")

tweets_df = pd.read_csv(filepath, encoding='utf-8')

weather_tweets = tweets_df[tweets_df['Content'].str.contains('climate', case=False, na=False)]

filtered_filename = 'weather_tweets.csv'
filtered_filepath = os.path.join(directory, filtered_filename)
weather_tweets.to_csv(filtered_filepath, index=False)

for content in weather_tweets['Content']:
    print(content)
    print("-" * 40)
