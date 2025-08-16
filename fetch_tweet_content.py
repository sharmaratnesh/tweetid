#This is a basic code for fetching tweet content based on list of tweetid
#Please update the bearer token and list of tweet id before using this script
#Cite this repository if you modify and use this piece of code.
#X(twitter) terms of use apply.
import requests
import json
import os
from urllib.parse import urlparse
import urllib3
from kaggle_secrets import UserSecretsClient

# Instantiate the client
secrets = UserSecretsClient()

# Retrieve the secret value
bearer_token = secrets.get_secret("<YOUR_TWITTER_BEARER_TOKEN_HERE")

# Disable SSL warnings for corporate environments if required
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_tweet_with_api_v2(tweet_id, bearer_token):
    """
    Fetch tweet content using Twitter API v2 (basic: only creation date and text)
    """
    url = f"https://api.twitter.com/2/tweets/{tweet_id}"
    params = {
        'tweet.fields': 'created_at,lang',
    }
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json'
    }
    try:
        response = requests.get(url, headers=headers, params=params, verify=False, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

import pandas as pd
def save_tweets_to_excel(tweet_list, filename="/kaggle/working/tweet_content.xlsx"):
    df = pd.DataFrame(tweet_list)
    df.to_excel(filename, index=False)

def main():
    # List of tweet IDs to fetch (example)
    tweet_ids = [
        <PUT YOUR LIST OF TWEET IDs HERE>
    ]

    tweets_data = []
    for tweet_id in tweet_ids:
        tweet_data = get_tweet_with_api_v2(tweet_id, bearer_token)
        if tweet_data and 'data' in tweet_data:
            tweet = tweet_data['data']
            tweet_text = tweet.get('text', '')
            tweet_lang = tweet.get('lang', '')
            print(f"Tweet text for {tweet_id}: {tweet_text}")
            print(f"Language: {tweet_lang}")
            tweets_data.append({
                'tweet_id': tweet.get('id', ''),
                'created_at': tweet.get('created_at', ''),
                'text': tweet_text,
                'lang': tweet_lang
            })
        else:
            print(f"Failed to fetch tweet ID: {tweet_id}")

    if tweets_data:
        save_tweets_to_excel(tweets_data)
        print(f"Saved {len(tweets_data)} tweets to Excel.")
    else:
        print("No tweets fetched.")

if __name__ == "__main__":
    main()
