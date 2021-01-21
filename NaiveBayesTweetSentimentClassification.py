import nltk
from twython import Twython


def bagOfWords(tweets):
    wordsList = []
    for (words, sentiment) in tweets:
        wordsList.extend(words)
    return wordsList

def getwordFeatures(wordList):
    wordList = nltk.FreqDist(wordList)
    wordFeatures = wordList.keys()
    return wordFeatures

def getFeatures(doc):
    docWords = set(doc)
    feat = {}
    for word in wordFeatures:
        feat['contains(%s)' % word] = (word in docWords)
    return feat



positiveTweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

negativeTweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

tweets = []
for (words, sentiment) in positiveTweets + negativeTweets:
    words_filtered = [e.lower() for e in nltk.word_tokenize(words) if len(e) >= 3]
    tweets.append((words_filtered, sentiment))

for t in tweets:
    print(t)

wordFeatures = getwordFeatures(bagOfWords(tweets))

training_set = nltk.classify.apply_features(getFeatures, tweets)

classifier = nltk.NaiveBayesClassifier.train(training_set)

print(classifier.show_most_informative_features(32))

ConsumerKey  = ""
ConsumerSecret = ""
AccessToken = ""
AccessTokenSecret = ""
 
twitter = Twython(ConsumerKey,
                  ConsumerSecret,
                  AccessToken,
                  AccessTokenSecret)

queryText = "python"
result = twitter.search(q=queryText)

for status in result["statuses"]:
    print("Tweet: {0} \n Sentiment: {1} \n".format( status["text"], classifier.classify(getFeatures(status["text"].split()))))

