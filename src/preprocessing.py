import pandas as pd

def parse_tweets(tweets):
    data = []
    for tweet in tweets:
        user, tweet_text = tweet.split('--#######--')
        data.append({'user': user.strip(), 'tweet': tweet_text.strip()})
    return data

if __name__ == '__main__':
    # Load the text data
    with open('src/data/tweets.txt', 'r') as f:
        tweets = f.readlines()
    
    # Parse the tweets
    tweet_data = parse_tweets(tweets)
    
    # Create a DataFrame
    df = pd.DataFrame(tweet_data)
    
    # Display the DataFrame
    print(df)

    # Save the DataFrame to a CSV file
    df.to_csv('src/data/tweets.csv', index=False)

    # Save only the tweet text to a text file
    with open('src/data/tweets_text.txt', 'w') as f:
        for tweet in tweet_data:
            f.write(tweet['tweet'] + '\n')