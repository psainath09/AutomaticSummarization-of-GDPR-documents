# import wrapper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# import cleaning and preprocessing libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.snowball import SnowballStemmer

# Text analysis helper libraries for word frequency
from nltk.tokenize import sent_tokenize

# for plotting the results
plt.style.use('ggplot')
import sklearn

#for model evaluation with the lda package

from tmtoolkit.lda_utils import tm_lda
from tmtoolkit.lda_utils.common import results_by_parameter
from tmtoolkit.lda_utils.visualize import plot_eval_results

## getting all scraped articles into dataframe
list_=[]
frame=pd.DataFrame()
for i in range(1,99):
    if os.path.getsize("Art. "+str(i)+" GDPR.tsv") > 0:
        data_content=pd.read_csv("Art. "+str(i)+" GDPR.tsv",sep="\t",lineterminator='\r',header=None,quoting=3,encoding = "ISO-8859-1")
        list_.append(data_content)
frame=pd.concat(list_,ignore_index=True)
frame["Article"]=frame.index+1
frame.columns=["Content","Article"]

# converting dataframe into list for further preprocessing convenience
doc_complete=frame.Content.tolist()
## function to clean the articles before building term document matrix
stemmer = SnowballStemmer("english")
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# tokenize and stem
def clean(article):
    article=" ".join([i for i in article.lower().split() if i not in set(stopwords.words("english"))])
    tokens=[word for sentence in nltk.sent_tokenize(article) for word in nltk.word_tokenize(sentence)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems=[lemma.lemmatize(s) for s in filtered_tokens]
    return stems

article_id=5

## build document term - inverse document frequency matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

tfidf=TfidfVectorizer(tokenizer=clean,stop_words='english', decode_error='ignore')
tdm=tfidf.fit_transform([frame["Content"][article_id]])
print("Built term frequency matrix!!")

## generating summary based on word importance

feature_names=tfidf.get_feature_names()

import math
from random import randint
from __future__ import division
#article_id= randint(0, tdm.shape[0] - 1)
article_text = frame["Content"][article_id]

sent_scores=[]
## calculating sentence importance based on word frequency
for sentence in nltk.sent_tokenize(article_text):
    score=0
    sent_tokens= clean(sentence)
    for token in (t for t in sent_tokens if t in feature_names):
        score += tdm[0,feature_names.index(token)]
    sent_scores.append((score/len(sent_tokens),sentence))
    
##declaring length of summary
summary_length = int(math.ceil(len(sent_scores) / 5))
sent_scores.sort(key=lambda sent: sent[0])
text=frame["Content"][article_id]
summary=""
for summary_sentence in sent_scores[:summary_length]:
    summary =summary + summary_sentence[1]
print(text)


print("summary content------")


print(summary)
## Gensim summarization using TextRank ##graphbase
from gensim.summarization import summarize
from gensim.summarization import keywords



reference=summarize(text)
print(reference)

## evaluation

from pythonrouge.pythonrouge import Pythonrouge

rouge = Pythonrouge(summary_file_exist=False,
                    summary=summary, reference=reference,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                    recall_only=True, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=50,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
#print(rouge.calc_score())

 # one word different
from nltk.translate.bleu_score import sentence_bleu

score = sentence_bleu(reference, summary)
print(score)



## Identifying topics using LDA for specific article
#doc_clean = [clean(doc).split() for doc in doc_complete] 
doc_clean = [clean(doc_complete[article_id])]
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

vect=CountVectorizer(ngram_range=(1,1))
dtm= vect.fit_transform(doc_clean[0])

pd.DataFrame(dtm.toarray(),columns=vect.get_feature_names())

lda= sklearn.decomposition.LatentDirichletAllocation()
lda_dtf=lda.fit_transform(dtm)

sorting=np.argsort(lda.components_)[:,::-1]
features=np.array(vect.get_feature_names())

import mglearn
mglearn.tools.print_topics(topics=range(10),feature_names=features,sorting=sorting,topics_per_chunk=5,n_words=10)


#topic_2 = np.argsort(lda_dtf[:,2])[::-1]
#for i in topic_1[:4]:
 #   print(" ".join(doc_clean[i])+ ".\n")
    
        
#LDA for summary generated #this code can be ignored if topics to be identified actual article content
#doc_clean_s = [clean(summary).split()]
#from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#from sklearn.decomposition import LatentDirichletAllocation
#import sklearn
#from gensim import corpora

#vect_s=CountVectorizer(ngram_range=(1,1))
#dtm_s= vect_s.fit_transform(doc_clean_s[0])

#pd.DataFrame(dtm_s.toarray(),columns=vect_s.get_feature_names())

#lda_s= sklearn.decomposition.LatentDirichletAllocation()
#lda_dtf_s=lda_s.fit_transform(dtm_s)

#sorting=np.argsort(lda_s.components_)[:,::-1]
#features_s=np.array(vect_s.get_feature_names())

#mglearn.tools.print_topics(topics=range(5),feature_names=features_s,sorting=sorting,topics_per_chunk=5,n_words=10)

## parameters for evaluation methods

const_params = dict(n_iter=20)
ks = list(range(1, 10, 10)) + list(range(1, 10, 20)) + list(range(30, 10, 50)) + [10, 20, 30]
varying_params = [dict(n_topics=k, alpha=1.0/k) for k in ks]

eval_results=[]
eval_results = tm_lda.evaluate_topic_models(dtm,
    varying_params,
    const_params)

results_by_n_topics = results_by_parameter(eval_results, 'n_topics')

plot_eval_results(results_by_n_topics, xaxislabel='numof topics k',title='Evaluation results for alpha=1/k, beta=0.1', figsize=(8, 6))
## results visualization

plt.tight_layout()
plt.show()

## visualization of topics

#from __future__ import print_function    
#import pyLDAvis
#import pyLDAvis.sklearn
#zit=pyLDAvis.sklearn.prepare(lda,dtm,vect)
#pyLDAvis.show(zit)




## end of project
