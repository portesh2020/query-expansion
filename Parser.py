import re
from PorterStemmer import PorterStemmer


class Parser:
  def __init__(self):
    self.porterStemmer = PorterStemmer()
    self.stopwords = set([line.strip() for line in open('stopwords.txt')])


  # return a list of terms from a string - special characters and stopwords removed
  def getTerms(self, str):
    rawTerms = [rawTerm.lower() for rawTerm in re.split("\s|(?<!\d)[^\w']+|[^\w']+(?!\d)", str)]
    terms = []

    for rawTerm in rawTerms:
      if rawTerm in self.stopwords or rawTerm == "":
        continue
      terms.append(rawTerm)

    return terms


  # return the stem word for a term
  def getStem(self, term):
    return self.porterStemmer.stem(term, 0, len(term) - 1).lower()


  # compare if two terms have the same stem words
  def hasSameStem(self, term1, term2):
    stem1 = self.getStem(term1)
    stem2 = self.getStem(term2)
    return (stem1 == stem2) or (term1 in term2) or (term2 in term1)