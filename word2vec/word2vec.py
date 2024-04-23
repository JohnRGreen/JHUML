# Import packages
# import jax.numpy as jnp
# from jax import random, vmap
# from jax.nn import embedding
# from jax.example_libraries import stax, optimizers, serial, Embedding, Flatten, Dense, Relu
import re

# set the working directory
import os
current_directory = os.getcwd()
print("Current working directory:", current_directory)
os.chdir('C:/Users/johng/OneDrive - Johns Hopkins/JHUML')

#### SET UP THE CORPUS ####
# Read the text corpus from a .txt file
file_path = "stevena.txt"
file_path = "debreu.txt"
with open(file_path, "r", encoding="utf-8") as file:
    corpus = file.read()

# preprocess
corpus = corpus.replace("\\n\\n", "")  # Remove "\" from the corpus
words = re.findall(r'\b\w+\b', corpus.lower())

# set length for n-grams; first hyper-parameter
n = 5
# Generate n-grams
n_grams = [words[i:i+n] for i in range(len(words) - n + 1)]

# input-output pairs
X = []
y = []
for n_gram in n_grams:
    X.append(n_gram[:-1])  # Input: First n-1 words
    y.append(n_gram[-1])   # Output: Last word

# Define vocabulary
vocab = sorted(set(words))
word_to_idx = {word: i for i, word in enumerate(vocab)}
vocab_size = len(vocab)

# Convert words to numerical indices
X_idx = [[word_to_idx[word] for word in n_gram] for n_gram in X]
y_idx = [word_to_idx[word] for word in y]
#### END CORPUS SETUP ####

#### SET UP THE NET ####
# import some packages for the NN
import numpy as np
from sklearn.model_selection import train_test_split
# For the NN we will use tensor flow
import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import Embedding, Flatten, Dense
from tf.keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

# Set the default float type
# keras.backend.set_floatx('float32')
tf.keras.backend.set_floatx('float32')

# Convert data to numpy arrays
X_train = np.array(X_idx)
y_train = np.array(y_idx)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define vocabulary size and embedding dimension
vocab_size = len(vocab)
embedding_dim = 100  # Adjust as needed

# Define the neural network architecture
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    Flatten(),
    Dense(units=vocab_size, activation='softmax')
])

# Define the optimizer with a custom learning rate
custom_lr = 0.025  # Adjust this value as needed
optimizer = Adam(learning_rate=custom_lr)

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
# We will train for 20 epochs
# and a batch size of 64 for the stochasti gradient descent
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))
#### END NN TRAINING ####

#### MAKE A PREDICTION ####
# Let's test the model out; give it a sentence:
sentence = "The New York Knicks are"
sentence = "Then an economy is"
words = sentence.lower().split()  # Tokenize the sentence and convert to lowercase
word_indices = [word_to_idx.get(word, 0) for word in words]  # Convert words to numerical indices
X_new_data = np.array(word_indices)

# Pad or truncate the input sequence to have a length of 4
max_sequence_length = 4
X_new_data_padded = pad_sequences([X_new_data], maxlen=max_sequence_length)
predictions = model.predict(X_new_data_padded)
# what is the probability of the max?
prob = np.max(predictions, axis=1)
print("Probability of the predicted word:", prob)
# Get the prediction weights on each word:
predicted_classes = np.argmax(predictions, axis=1)
# What is this from our dirctionary?
predicted_word = [vocab[idx] for idx in predicted_classes]
print("The predicted word is:", predicted_word)

# Append the predicted word to the sentence
sentence += " " + predicted_word[0]

# Update the n-gram with the new word
n_gram = sentence.split()[-n:]
X_new_data = [word_to_idx.get(word, 0) for word in n_gram]
X_new_data_padded = pad_sequences([X_new_data], maxlen=max_sequence_length)

# Continue predicting for the next word
predictions = model.predict(X_new_data_padded)
predicted_classes = np.argmax(predictions, axis=1)
predicted_word = [vocab[idx] for idx in predicted_classes]

# If you want to map the predicted classes back to words in your vocabulary
predicted_word = [vocab[idx] for idx in predicted_classes]
sentence += " " + predicted_word[0]
#### END PREDICTION ####


#### DEBREU BOT ####

#### END DEBREU BOT ####


#### SCRAPE TEXT ####
import requests
from bs4 import BeautifulSoup

# Function to scrape and save article text
def scrape_article(url, output_dir):
    # Fetch the HTML content of the article page
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        # Extract article text
        article_text = soup.find('div', class_='article-body').get_text()
        # Save article text to a txt file
        filename = os.path.join(output_dir, url.split('/')[-1] + '.txt')
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(article_text)
    else:
        print(f"Failed to fetch article from {url}")

# Main function to scrape all articles
def scrape_all_articles(base_url, output_dir):
    # Fetch the HTML content of the base URL
    response = requests.get(base_url)
    if response.status_code == 200:
        html_content = response.text
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        # Find all links to articles
        article_links = soup.find_all('a', class_='story-link')
        # Iterate through each article link and scrape its content
        for link in article_links:
            article_url = link['href']
            # Ensure the link is complete
            if not article_url.startswith('http'):
                article_url = 'http://www.espn.com' + article_url
            scrape_article(article_url, output_dir)
    else:
        print("Failed to fetch base URL.")

if __name__ == "__main__":
    base_url = "http://www.espn.com/new-york/columns/archive?name=stephen-a-smith"
    output_directory = "articles"  # Directory to save scraped articles
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    scrape_all_articles(base_url, output_directory)

if __name__ == "__main__":
    base_url = "http://www.espn.com/new-york/columns/archive?name=stephen-a-smith"
    output_directory = "articles"  # Directory to save scraped articles
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    scrape_all_articles(base_url, output_directory)

# URL of the ESPN page containing links to Steven A Smith articles
url = "http://www.espn.com/new-york/columns/archive?name=stephen-a-smith"
url = "https://www.project-syndicate.org/columnist/slavoj-zizek"
url = "https://www.happyscribe.com/public/the-bill-simmons-podcast"
# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.content, "html.parser")

# Find all episode containers
episode_containers = soup.find_all("a", class_="hsp-card-episode")

# Function to scrape text from episode link
def scrape_episode_text(episode_url):
    response = requests.get(episode_url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Find the element containing the text you want to scrape
    episode_text = soup.find("div", class_="your-text-class").text.strip()
    return episode_text

# Loop through each episode container and extract relevant information
for episode in episode_containers:
    episode_title = episode.find("h3", class_="hs-font-positive base bold").text.strip()
    episode_link = episode['href']
    podcast_name = episode.find("h4", class_="hs-font-positive small-base").text.strip()
    view_count = episode.find("li", class_="view-count").text.strip()
    upload_date = episode.find("li", class_="date").text.strip()
    duration = episode.find("li", class_="duration").text.strip()
    description = episode.find("p", class_="hsp-card-episode-description").text.strip()

    print("Episode Title:", episode_title)
    print("Podcast Name:", podcast_name)
    print("View Count:", view_count)
    print("Upload Date:", upload_date)
    print("Duration:", duration)
    print("Description:", description)
    
    # Scrape text from the episode link
    episode_text = scrape_episode_text(episode_link)
    print("Episode Text:", episode_text)
    
    print()

# Find all the links to Steven A Smith articles
article_links = soup.find_all("a", class_="article-link")

print(soup.prettify()[:5000])  # Change the number 1000 as per your requirement
