from bs4 import BeautifulSoup
import re
import csv
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
    
#we're going to load in our test and training data 
#test = pd.read_csv('testtrolls.csv', header=True, names=['y', 'date', 'Comment'])
#training = pd.read_csv('trainingtrolls.csv', header=True, names=['y', 'date', 'Comment', 'Usage'])

    
def cleaner(inputfile):
    trolls = []
    for line in inputfile:
        
        line = BeautifulSoup(str(line["Comment"]), "html.parser")  
        trolls.append(line)
        line = re.sub(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', '_EM', str(line))
        
        line = re.sub(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', '_EM', str(line))
        
        line = re.sub(r'\w+:\/\/\S+', r'_U', line)
        line = line.replace('"', ' ')
        line = line.replace('\'', ' ')
        line = line.replace('_', ' ')
        line = line.replace('-', ' ')
        line = line.replace('\n', ' ')
        line = line.replace('\\n', ' ')
        line = line.replace('\'', ' ')
        line = re.sub(' +',' ', line)
        line = line.replace('\'', ' ')

        line = re.sub(r'([^!\?])(\?{2,})(\Z|[^!\?])', r'\1 _BQ\n\3', line)
        line = re.sub(r'([^\.])(\.{2,})', r'\1 _SS\n', line)
        line = re.sub(r'([^!\?])(\?|!){2,}(\Z|[^!\?])', r'\1 _BX\n\3', line)
        line = re.sub(r'([^!\?])\?(\Z|[^!\?])', r'\1 _Q\n\2', line)
        line = re.sub(r'([^!\?])!(\Z|[^!\?])', r'\1 _X\n\2', line)
        line = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2 _EL', line)
        line = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2 _EL', line)
        line = re.sub(r'(\w+)\.(\w+)', r'\1\2', line)
        
        #Swears
        line = re.sub(r'([#%&\*\$]{2,})(\w*)', r'\1\2 _SW', line)
        line = re.sub('[1|2|3|4|5|6|7|8|9|0]', '', line)
        # big and happy smileys       
        line = re.sub(r' [8x;:=]-?(?:\)|\}|\]|>){2,}', r' _BS', line)
        # small and happy smileys       
        line = re.sub(r' (?:[;:=]-?[\)\}\]d>])|(?:<3)', r' _S', line)
        #big and sad smileys
        line = re.sub(r' [x:=]-?(?:\(|\[|\||\\|/|\{|<){2,}', r' _BF', line)
        #small and sad
        line = re.sub(r' [x:=]-?[\(\[\|\\/\{<]', r' _F', line)
        line = re.sub('[%]', '', line)
        #print(line)
        
        phrases = re.split(r'[;:\.()\n]', line)
        phrases = [re.findall(r'[\w%\*&#]+', ph) for ph in phrases]
        phrases = [ph for ph in phrases if ph]

        words = []

        for ph in phrases:
            words.extend(ph)
        #print(words)
        
        tmp = words
        words = []
        new_word = ''
        for word in tmp:
            if len(word) == 1:
                new_word = new_word + word
            else:
                if new_word:
                    words.append(new_word)
                    new_word = ''
                words.append(word)
        #print(words)

        words = [w for w in words if not w in stopwords.words("english")]
        #print(words)

        tagged = []
        #nltk.download('averaged_perceptron_tagger')     
        #nltk.download('wordnet')
        for t in words:
            t = t.lower()
            treebank_tag = pos_tag([t])
            tagged.append(treebank_tag)
        #print(tagged)

         #this function just translates between the PoS tags used by our Treebank tagger and the WordNet equivalents.     
        #As it was trained with the Treebank corpus, it also uses the Treebank tag set.


        def get_wordnet_pos(treebank_tag):

            if treebank_tag[0][1].startswith('J'):
                return wordnet.ADJ
            elif treebank_tag[0][1].startswith('V'):
                return wordnet.VERB
            elif treebank_tag[0][1].startswith('N'):
                return wordnet.NOUN
            elif treebank_tag[0][1].startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN
        postagged = []
        for t in tagged:
            newtag = t[0][0],get_wordnet_pos(t)
           # print(get_wordnet_pos(t))
            postagged.append(newtag)
        #print(postagged) 
        
        stemmer = PorterStemmer()
        #print(words)
        stemmed_words = []
        for w in words:
            stemmed_words.append(stemmer.stem(w))
       # print(stemmed_words)

        lemmatizer = WordNetLemmatizer()
        lemmatized_words =[]
        for w in words:
            lemmatized_words.append(lemmatizer.lemmatize(w))
        #print(lemmatized_words)

        lemmatized = []
        for t in postagged:
            lemmatized.append(lemmatizer.lemmatize(t[0], t[1]))
        #print(lemmatized) 
        
        trolls.append(lemmatized)
        #print(trolls)
        return(trolls)
    
with open('trainingtrolls.csv',  'rt') as f:
    train_reader = csv.DictReader(f)    
    train_x = cleaner(train_reader)
print(train_x[1])

with open('testtrolls.csv','rt') as f:
    test_reader = csv.DictReader(f)
    test_x = cleaner(test_reader)
vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,  max_features = 5000)

train_data_features = []

for line in train_x:
    train_data_features.append(vectorizer.fit_transform(line))
print(train_data_features[0])