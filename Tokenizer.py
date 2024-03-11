from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem.porter import *

class Tokenizer:

    def __init__(self):
        english_stopwords = frozenset(stopwords.words('english'))
        
        corpus_stopwords = ['people', 'second', 'history', 'external', 'first', 'see', 'my', "you're",
                    'became', 'one', 'ourselves', 'i', 'university', 'we', 'many', 'new', 'two',
                    'me', 'district', 'however', 'references', 'ours', 'thumb', 'our', 'category',
                    'you', 'would', 'part', 'links', 'myself', 'house', 'may', 'also', 'following', 'including']
        
        self.all_stopwords = english_stopwords.union(corpus_stopwords)
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        self.porter_stemmer = PorterStemmer()
    # Getting tokens from the text while removing punctuations.
    def filter_tokens(self, tokens, tokens2remove=None, use_stemming=False):
        ''' The function takes a list of tokens, filters out `tokens2remove` and
        stem the tokens using `stemmer`.
        Parameters:
        -----------
        tokens: list of str.
            Input tokens.
        tokens2remove: frozenset.
            Tokens to remove (before stemming).
        use_stemming: bool.
            If true, apply stemmer.stem on tokens.
        Returns:
        --------
        list of tokens from the text.
        '''
        stemmed_tokens = []
        if tokens is not None:
            if tokens2remove is not None:
                for token in tokens:
                    if token not in self.all_stopwords:
                        if use_stemming:
                            token = self.porter_stemmer.stem(token)
                        stemmed_tokens.append(token)

        if use_stemming :
            res = []
            for x in stemmed_tokens :
                stem = self.porter_stemmer.stem(x)
                if not stem in res:
                    res.append(stem)

        return stemmed_tokens
    def tokenize(self, text, use_stemm=False):
        return self.filter_tokens([token.group() for token in self.RE_WORD.finditer(text.lower())],self.all_stopwords, use_stemm)