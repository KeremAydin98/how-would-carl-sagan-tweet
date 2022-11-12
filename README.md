# how-would-have-carl-sagan-tweeted

I love Carl Sagan, he is one of the first rockstar scientists that sparked the interest of ordinary people like me to the science and astronomy. He has died before I was even born, but he still thought and inspired me with his books. And while I was reading one his predecessor Neil Degrasse Tyson's tweets, I wondered how would have Carl Sagan tweeted if he was alive in the social media era? I do not need to wonder in a time where robots complete the unfinished symphony of Beethoven. Therefore, I constructed a language model to predict Carl Sagan tweets using his books as a train dataset. 

I used two of Carl Sagan books which are named Cosmos and Billions & billions of thoughts and converted them to text files from pdfs using PyPdf2 library and cleaned the wrong extractions by hand.

Train data set for a text generation model is gathered by using a sliding window. Let’s say our window size is 6. Firstly we move the window from left to right on our text. On the window first five characters are used as features and last character as target or label. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/198904154-f92cde1e-d0b0-42bd-96a1-39ad55b6175d.png" />
</p>

With the sliding window technique, I formed the train dataset from the text files. Text generation model trains itself by predicting the target or label character with the feature characters. The model consists of RNN structures since the information of sequence must be preserved to predict the next token. RNN structures preserve some kind of memory in one of their outputs called hidden state, then the next RNN cell uses the input word and hidden state as the information of the previous characters to predict the next character. Actually it does not predict the next character, it only predicts the probability of the next character using linear layer then a softmax layer.

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/198904289-81b86395-f14e-42f1-9701-fa8fdaa731cd.png" />
</p>

After generating texts I realised some spelling errors in the generated text since the model is generating text one character at a time. Therefore, to correct the spelling errors TextBlob library was utilized.

And finally, I wanted to create some sort of hashtags for the tweets. Since the hashtags are usually the keywords of the tweet, I used KeyBERT as a keyword extraction model. KeyBERT replaces words with the BERT embedding vectors and compares the embedding of the whole body of the text with each word's embedding vector to find which one is the most similar. That's basic idea at least. 

Some of the tweets I generated can be seen below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/198904049-39a1faf7-5aa2-417a-91b9-08b89151a2ce.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/198904058-4128eb4c-0b48-4bcd-9948-7c60470b931e.png" />
</p>

They make sense in some way, but it can still be improved using more layers and more data. I was limited by my GPU capability.
