import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

language = 'english'
ps = PorterStemmer()
nltk.download('stopwords')
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

# Clean HTML tags from text
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

# Clean text : remove HTML tags, remove non alphabetic characters, set to lower case, split in words, stem words, rebuil the sentence and return it
def cleantext(sentence):
    sentence = cleanhtml(sentence)
    sentence = re.sub('[^a-zA-Z]',' ', sentence)
    sentence = sentence.lower()
    sentence = sentence.split()
    # Stemming is the process of producing morphological variants of a root/base word
    sentence = [ps.stem(word) for word in sentence if not word in all_stopwords] 
    sentence = ' '.join(sentence)
    return sentence