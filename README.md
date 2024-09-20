# Political Tweet Classification 

## Set-up

### Local

1. Set up the environment:
    * `make venv`
    * `make install`

2. Run the project
    * `source ./.venv/bin/activate`
    * `python src/main.py`

### Docker

1. Build and run the Docker container
    * `make build`
    * `make run`

## Idea 

### Data

The folder [data](/src/data/) contains datasets used to train and test our ML Text Classification Process.
* [tweets.csv](/src/data/tweets.csv) is our dataset containing tweets with the respective user who posted them. 
* [tweets_text.txt](/src/data/tweets_text.txt) only contains the text of each tweet independently from the user.
* [labeled_tweets.txt](/src/data/labeled_tweets.txt) is a first attempt to classify to which candidate the tweet refers. 

### ML

1. Identify the candidate to which the tweet refers.
    * **Naive approach**: `if "Trump" in tweet`
    * **ML approach**: train a text classification model to label the tweet: `__label__trump`, `__label__harris`
        * First attempt: 
            1. Classified each tweet with ChatGPT 
            2. Trained fastText `fasttext.train_supervised`. Check it [here](/src/train.py).
2. Assess whether the message, referring to either one of the candidate, is positive or negative.
    * Check the [file](/src/main.py).
    * For the experiment it has been used the base model [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest),
    * Which is easily viable trough the Python3 library [tweetnlp](https://github.com/cardiffnlp/tweetnlp).
    * However a more fine-tuned version of the model towards politic tweets could be used. Check it [here](https://huggingface.co/cardiffnlp/xlm-twitter-politics-sentiment).
3. Increase the score of the candidate with a positive tweet. Decrease the score of a candidate with negative tweet.  
