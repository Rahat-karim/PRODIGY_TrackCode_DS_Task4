import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv('twitter_training.csv', header=None)
valid_df = pd.read_csv('twitter_validation.csv', header=None)

# Combine training and validation datasets
df = pd.concat([train_df, valid_df])

# Set the column names manually based on the dataset structure
df.columns = ['ID', 'Entity', 'Sentiment', 'Content']

# Ensure all entries in 'Content' are strings
df['Content'] = df['Content'].astype(str)

# Function to clean tweet text
def clean_tweet(tweet):
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  # Remove @ mentions
    tweet = re.sub(r'#', '', tweet)  # Remove '#' symbols
    tweet = re.sub(r'RT[\s]+', '', tweet)  # Remove RT (retweets)
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)  # Remove hyperlinks
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Remove punctuation
    tweet = tweet.lower()  # Convert to lowercase
    return tweet

# Apply the cleaning function to the tweets
df['cleaned_tweet'] = df['Content'].apply(clean_tweet)

# Function to get the sentiment
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Apply the sentiment function
df['sentiment'] = df['cleaned_tweet'].apply(get_sentiment)

# Count the sentiment values
sentiment_counts = df['sentiment'].value_counts()

# Plot the sentiment distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Time-Based Sentiment Analysis (Optional if timestamp exists)
if 'timestamp' in df.columns:
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Resample by month and count sentiments
    monthly_sentiment = df.resample('M')['sentiment'].value_counts().unstack().fillna(0)

    # Plot the time series sentiment
    plt.figure(figsize=(14, 7))
    monthly_sentiment.plot(kind='line', marker='o')
    plt.title('Sentiment Over Time')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.legend(title='Sentiment')
    plt.show()