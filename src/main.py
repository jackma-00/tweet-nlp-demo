import tweetnlp
import pandas as pd


# load model for sentiment analysis
model = tweetnlp.load_model('sentiment')


def inference(text):
    """
    Perform sentiment analysis on the given text.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing the sentiment label and its probability.
    """
    label = model.sentiment(text, return_probability=True)
    return label


if __name__ == '__main__':

    # Load the text data
    with open('src/data/tweets_text.txt', 'r') as f:
        tweets = f.readlines()
        for tweet in tweets:
            tweet = tweet.replace("\n", "").strip()
            # Perform sentiment analysis
            result = inference(tweet)
            print(f"Text: {tweet}")
            print(f"Sentiment: {result['label']}")
            print(f"Probability: {result['probability']}")
            print()
            


    
    