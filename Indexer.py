import re
from Parser import Parser


class Indexer:
  def __init__(self):
    self.relevantDocs = dict()
    self.nonRelevantDocs = dict()
    self.invertedFiles = dict()


  # store a document given relevance feedback from the user
  def insertDoc(self, url, docVector, isRelevant):
    if isRelevant:
      self.relevantDocs[url] = docVector
    else:
      self.nonRelevantDocs[url] = docVector


  # update inverted files given a document
  def insertInvertedFiles(self, doc):
    parser = Parser()
    terms = parser.getTerms(doc["title"] + doc["snippet"])

    for term in terms:
      if term not in self.invertedFiles:
        self.invertedFiles[term] = {}

      if doc["link"] not in self.invertedFiles[term]:
        self.invertedFiles[term][doc["link"]] = 1
      else:
        self.invertedFiles[term][doc["link"]] += 1


  # clear out all indexing
  def clearIndex(self):
    self.relevantDocs = dict()
    self.nonRelevantDocs = dict()
    self.invertedFiles = dict()