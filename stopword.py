import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
stop = stopwords.words('english')

# delete numbers
import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))
 
# delete symbol
def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))
 
# lemma
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
 
def check(word):
 #check element 
    word= word.lower()
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True
 
# process words
def preprocessing(sen):
    res = []
    for word in sen:
        if check(word):
             if len(word)>1:
                #delete useless words
                word = word.lower()
                #if this element satisfied requirement, append to list
                res.append(wordnet_lemmatizer.lemmatize(word))
    return res            
