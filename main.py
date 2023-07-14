import itertools
import nltk
import numpy as np
import re
import sys
from collections import Counter
from GoogleApiClient import GoogleApiClient
from Indexer import Indexer
from itertools import permutations
from lib2to3.pgen2 import token
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import word_tokenize, sent_tokenize
from nltk.lm import MLE
from Parser import Parser
nltk.download('punkt')


# neatly print a document
def printDoc(i, doc):
  print("\nDocument " + str(i))
  print("[")
  print("\tTitle:\n\t" + doc["title"])
  print("\n\tURL:\n\t" + doc["link"])
  print("\n\tSnippet:\n\t" + doc["snippet"])
  print("]\n")


# ask user input on a document
def isRelevant(i, doc):
  printDoc(i, doc)

  while True:
    userInput = input("Is this document relevant? (y/n)\n").lower()

    if userInput == "y":
      return True
    elif userInput == "n":
      return False
    else:
      print("Enter 'y/Y' or 'n/N' only")
      continue


# vectorize the query string
def getQueryVector(invertedFiles, query):
  parser = Parser()
  counter = Counter(parser.getTerms(query))
  queryVector = dict()

  for term in invertedFiles:
    if term in counter:
      tf = counter[term]
      idf = 10 / len(invertedFiles[term].keys())
      queryVector[term] = (1 + np.log10(tf)) * np.log10(idf)
    else:
      queryVector[term] = 0

  return queryVector


# vectorize a document
def getDocVector(invertedFiles, doc):
  docVector = dict()

  for term in invertedFiles:
    if doc["link"] in invertedFiles[term]:
      tf = invertedFiles[term][doc["link"]]
      idf = 10 / len(invertedFiles[term].keys())
      docVector[term] = (1 + np.log10(tf)) * np.log10(idf)
    else:
      docVector[term] = 0

  return docVector


# choose top two terms using Rocchio's algorithm
def rocchio(invertedFiles, relevantDocs, nonRelevantDocs, query):
  # extract terms and numerical values of the query vector
  corpus = np.array(list(getQueryVector(invertedFiles, query).keys()))
  queryVector = np.array(list(getQueryVector(invertedFiles, query).values()))

  # sum document vectors for each relevant and non-relevant documents
  relevantVectorSum = np.zeros(len(corpus))
  nonRelevantVectorSum = np.zeros(len(corpus))

  for doc in relevantDocs:
    relevantVector = np.array(list(relevantDocs[doc].values()))
    relevantVectorSum += relevantVector

  for doc in nonRelevantDocs:
    nonRelevantVector = np.array(list(nonRelevantDocs[doc].values()))
    nonRelevantVectorSum += nonRelevantVector

  # constants for Rocchio's algorithm
  alpha = 1
  beta = 0.75
  gamma = 0.15
  r = len(relevantDocs)
  nr = len(nonRelevantDocs)

  # update query vector using Roochio's algorithm
  newQueryVector = alpha * queryVector + beta * (relevantVectorSum / r) - gamma * (nonRelevantVectorSum / nr)

  # sort terms by tf-idf values
  finalFeedbackVector = {}

  for i in range(len(newQueryVector)):
    finalFeedbackVector[corpus[i]] = newQueryVector[i]

  sortedVectorByTFIDF = sorted(finalFeedbackVector.items(), key=lambda x:x[1], reverse=True)

  # choose top two terms that don't have the same stem word as any existing query term
  count = 0

  while count < 2:
    queryTerms = query.split(" ")

    for (term, tfidf) in sortedVectorByTFIDF:
      alreadyExist = False
      for queryTerm in queryTerms:
        if Parser().hasSameStem(term, queryTerm):
          alreadyExist = True
          break

      if not alreadyExist:
        if count == 0:
          newTerm1 = term
          query = query + " " + newTerm1
          count = 1
        else:
          newTerm2 = term
          count = 2
          break

  return (newTerm1, newTerm2)


def main():
  if len(sys.argv) != 5:
    print("Usage: <google api key> <search engine id> <precision> <query>")
    exit()

  googleApiClient = GoogleApiClient(sys.argv[1], sys.argv[2])
  target = float(sys.argv[3])
  query = sys.argv[4]
  indexer = Indexer()
  parser = Parser()

  while True:
    # retrieve results from Google Custom Search API
    docs = googleApiClient.getQueryResults(query)
    print("Query: ", query)

   # terminate the program if fewer than 10 documents found
    if len(docs) < 10:
      print("Program terminating... fewer than 10 results found")
      exit()

    # index retrieved documents into inverted files
    for doc in docs:
      indexer.insertInvertedFiles(doc)
    invertedFiles = indexer.invertedFiles

    # take user relevance feedback and store document vectors accordingly
    for i, doc in enumerate(docs):
      docVector = getDocVector(invertedFiles, doc)
      indexer.insertDoc(doc["link"], docVector, isRelevant(i, doc))
    relevantDocs = indexer.relevantDocs
    nonRelevantDocs = indexer.nonRelevantDocs

    # terminate the program if no relevant documents found
    if len(relevantDocs) == 0:
      print("Program terminating... no relevant documents found")
      exit()

    # compute precision and print results
    precision = len(relevantDocs) / 10
    print("\nTarget precision: ", target)
    print("Precision: ", precision)

    # break out of the loop if target precision met
    if precision >= target:
      print("Reached target precision! Done")
      break

    # otherwise, run Rocchio algorithm for choosing two new words to expand query
    top2Terms = rocchio(invertedFiles, relevantDocs, nonRelevantDocs, query)
    (newTerm1, newTerm2) = top2Terms

    # create bigram logic to choose term ordering of the new query - get all titles and snippets out
    content = " "
    for doc in docs:
      if doc["link"] in relevantDocs:
        title = doc["title"]
        snippet = doc['snippet']
        content += " " + title + " " + snippet

    # tokenize with built-in tokenizer
    tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(content)]
    n = 2
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    model = MLE(n)
    model.fit(train_data, padded_sents)

    # keep track of the sequence with the highest probability
    highestProb = {}

    # new query by adding two words returned by rocchio's algorithm
    query = query + " " + newTerm1 + " " + newTerm2
    queryPermutations = query.split(" ")

    # find probability of every possible ordering of terms in the query and pick the highest
    for perm in itertools.permutations(queryPermutations):
      score = 1
      for i in range(len(perm)-1):
        firstWord = perm[i].lower()
        secondWord = perm[i+1].lower()
        score *= model.score(secondWord, firstWord.split())
      highestProb[perm] = score

    # find the query ordering with the highest probability
    optimizedQuery = sorted(highestProb.items(), key=lambda x:x[1], reverse=True)
    query = " ".join(list(optimizedQuery[0][0]))

    # print new query and clear index
    print("Augmenting query by '" + newTerm1 + "' and '" + newTerm2 + "'")
    print("Final query with ordering: ", query, "\n")
    indexer.clearIndex()


if __name__ == "__main__":
    main()