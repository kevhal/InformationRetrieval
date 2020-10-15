import random
import codecs
import string
import re
import numpy as np
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

import gensim
from gensim import corpora

# 1.0
random.seed(123)
stemmer = PorterStemmer()


def preprocess(document):
    if (document != ""):
        tokenizedParagraphs = document.split(" ")

        # remove punctuation and set to lowercase
        processedTokens = [i.translate(str.maketrans('', '', string.punctuation)).lower() for i in tokenizedParagraphs]

        # stem words
        processedTokens = [stemmer.stem(token) for token in processedTokens]
        return processedTokens

    else:
        # 1.1
        f = codecs.open("pg3300.txt", "r", "utf-8")
        # 1.2
        paragraphs = f.read().split("\r\n\r\n")
        paragraphs = [paragraph.strip("\r\n") for paragraph in paragraphs]
        # 1.3
        paragraphs = [paragraph for paragraph in paragraphs if
                      not paragraph.__contains__("Gutenberg") and not paragraph.__contains__("gutenberg") and not paragraph.__contains__("GUTENBERG")]
        # 1.4
        tokenizedParagraphs = [re.split(" |\r\n", i) for i in paragraphs]
        #Remove
        processedTokens = [[j.replace("\r\n", "").lower() for j in i if not j == ""] for i in tokenizedParagraphs]

        # 1.5
        # remove punctuation
        processedTokens = [[j.translate(str.maketrans('', '', string.punctuation)) for j in i] for i in processedTokens]

        # 1.6
        # stem words
        processedTokens = [[stemmer.stem(j) for j in i] for i in processedTokens]
        return processedTokens, paragraphs


processedTokens, paragraphs = preprocess("")
# 2.1 Create dictionary
dictionary = corpora.Dictionary(processedTokens)
f = codecs.open("common-english-words.txt", "r", "utf-8")
stop_words = f.read().split(",")
ids = []
for stopword in stop_words:
    stopword = stemmer.stem(stopword)
    try:
        ids.append(dictionary.token2id[stopword])
    except:
        pass

dictionary.filter_tokens(ids)

# 2.2 Create corpus
corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in processedTokens]

# 1.7 Show each tokens frequency
word_counts = [[(dictionary[id], count) for id, count in doc] for doc in corpus]

# create models
# 3.1
tfidf_model = gensim.models.TfidfModel(corpus)

# 3.2
#Creates the tf-idf weights of each token
tfidf_corpus = tfidf_model[corpus]

# 3.3
#Creates similarity matrix
indexT = gensim.similarities.MatrixSimilarity(tfidf_corpus)

# 3.4
#Same as 3.2 and 3.3 just for a LSI-model
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
indexL = gensim.similarities.MatrixSimilarity(lsi_model[tfidf_corpus])

# 3.5
#Prints the top three LSI-topics
[print(f"{index} {topic}") for index, topic in lsi_model.print_topics(num_topics=3)]

# 4.1
rawQuery = "What is the function of money?"
#preprocesses the query
processedQuery = preprocess(rawQuery)
#Turn query tokens into Bag of Words
queryBoW = dictionary.doc2bow(processedQuery)
# 4.2
#Prints the weight of each token in the query
query_weights = tfidf_model[queryBoW]
s = ""
for i in range(len(processedQuery)):
    s += f"{processedQuery[i]}: {query_weights[i][1]}, "
s = s[:-1]
print(str(s + "\n"))
# 4.3
#Prints the first 5 lines of the top three matching paragraphs
doc2similarity = enumerate(indexT[tfidf_model[queryBoW]])
for doc_number, score in sorted(doc2similarity, key=lambda x: -x[1])[:3]:
    paragraph = "\r\n".join([word.replace("\n", "").replace("\r", "") for word in paragraphs[doc_number].split("\r\n")[:5]])
    print(f"[Paragraph {doc_number}, score: {score}]\n{paragraph}\n")

# 4.4
#Prints the top three topics for the query
lsi_query = lsi_model[queryBoW]
lsi_topics = lsi_model.show_topics(100)
for topicNum, score in enumerate(sorted(lsi_query, key=lambda x: -abs(x[1]))[:3]):
    print(f"Topic nr{score[0]}  Score: {score[1]}, Topic {lsi_topics[score[0]][1]}\n")

doc2similarity2 = enumerate(indexL[lsi_model[queryBoW]])
#Prints the top three paragraph according to the LSI-model
for doc_number, score in sorted(doc2similarity2, key=lambda x: -abs(x[1]))[:3]:
    paragraph = "\r\n".join(paragraphs[doc_number].split("\r\n")[:5])
    print(f"#{doc_number + 1} [Paragraph {doc_number}, score: {score}]\n{paragraph}\n")

#Graphs the 15 most frequently used words
test = []
for index in dictionary.id2token:
    if not ids.__contains__(index):
        test.append((dictionary[index], dictionary.cfs[index]))
test = sorted(test, key=lambda x: -x[1])[:15]
x, y = [word for word, freq in test], [freq for word, freq in test]

plt.figure(figsize=(15, 8))
plt.bar(np.arange(len(x)), y, align="center", alpha=0.5)
plt.xticks(np.arange(len(x)), x)
plt.xlabel("Processed tokens")
plt.ylabel("Frequency")
plt.title("Frequency distribution for top 15 most frequently used tokens")
plt.savefig("frequency_distributions.png")

