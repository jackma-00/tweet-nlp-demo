import fasttext

model = fasttext.train_supervised('src/data/labeled_tweets.txt')

with open('src/data/tweets_text.txt', 'r') as f:
    tweets = f.readlines()
    for tweet in tweets:
        tweet = tweet.replace("\n", "").strip()
        # Perform sentiment analysis
        result = model.predict(tweet)
        print(f"Text: {tweet}")
        print(f"Sentiment: {result[0][0]}")
        print(f"Probability: {result[1][0]}")
        print()