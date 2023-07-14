## Authors
Kate Jeon 
Hannah Portes 


## Files
Name | Purpose
--- | ---
``tests`` | Folder with test case results
``__init__.py`` | Namespace package (no content)
``GoogleApiClient.py`` | Client class that calls the Google Custom Search API
``Indexer.py`` | Indexer class that indexes documents fetched by the API
``main.py`` | Main program incorporating vectorizations and Rocchio's algorithm
``Parser.py`` | Parser class that handles all text-related functionalities
``PorterStemmer.py`` | [Public library](http://www.tartarus.org/martin/PorterStemmer) imported to extract the stem word of terms
``stopwords.txt`` | [Public list](https://gist.github.com/sebleier/554280) imported to remove stop words from query expansion


## Installation & Usage

### Dependencies
```
sudo apt install python3-pip # if pip3 is not installed
pip3 install nltk
pip3 install numpy --upgrade
```

### Program Run Instructions
In the root of the project directory run the following command:
```
$ python3 main.py <google api key> <search engine id> <precision> <query>
```

## Architectural Design
### Custom Classes
- ``GoogleApiClient``:
    This class implements the basic *getQueryResults* method to fetch the top ten results by calling Google's Custom Search API given an input query.

- ``Indexer``:
    Indexer stores documents retrieved from the API categorized by user relevance feedback accordingly. Additionally, it indexes all documents into inverted files as the sample data structure below. Note that url is used as a unique id for each document.
    ```
    {
      <term1>: {
                  <url1>: tf1,
                  <url2>: tf2
               },
      <term2>: {
                  <url3>: tf3
               },
      <term3>: {
                  <url4>: tf4,
                  <url5>: tf5
               },
        .
        .
        .

    }
    ```
    There are 3 functions - *insertDoc*, *insertInvertedFiles*, and *clearIndex*. *insertDoc* puts each document url and calculated vector into a dictionary of relevant or non-relevant documents, based off the feedback provided from the user. *insertInvertedFiles* creates a dictionary storing every term in the title and snippet of each of the documents retrieved and their counts (i.e., term frequency). This is utilized when calculating tf-idf vectors for each document. Lastly, *clearIndex* clears out all relevant and non-relevant documents, and the inverted files for new iteration.

- ``PorterStemmer``: This is an imported third party library released for the final time in 2008. It carries out an algorithm for suffix stripping to extract the stem of words (e.g., "fighter" and "fighting" share a same stem). This prevents words sharing the same stem from being added to the query. For example, once "fighter" has been added to the query, "fighting" will not be considered as a viable option to be added to the augmented query in future iterations. Detailed descriptions can be found [here](http://www.tartarus.org/martin/PorterStemmer).

- ``Parser``: Parser handles all text-related functionalities required for the program. It includes functions that preprocesses document data (by converting characters to lowecase and removing stopwords and non-alphabetical characters), extracts stem word from a term, and compares if two terms have the same stem word. The list of stopwords are in `stopwords.txt` and its reference link can be found [here](https://gist.github.com/sebleier/554280).


### Main program
The program starts by retrieving results from Google API using the query that was given from the user as one of the arguments. As it retrieves ten results from the API, we index data as inverted files and categorically save documents as we take in user relevance feedback. Url is used as an identifier for each document and thus stored as a key, and for its value, we store the document vector. The functions *getQueryTerms* and *getQueryVector* work in conjunction to vectorize the query string. We calculate the TFIDF value for present terms and terms that are not present are set to 0. We calculate each document vector similarly in *getDocVector*. Until the target precision is met, we add two words to the query using Rocchio's algorithm, determine ordering splitting the query into bigrams to find the most likely ordering, and run the next iteration. Edge cases such as less then ten results fetched or user marking zero relevant document are gracefully handled by terminating the program.


## Query Expansion Algorithm
There are two steps of query expansion. First, to select the two new words to extend the query, Rocchio's algorithm is implemented in *rocchio* (under `main.py`). This function takes all the files, separated into relevant and non-relevant documents, and the input query. Within the function, all relevant and non-relevant document vectors are summed so there is one remaining vector for each category. These sum of the relevant vectors are then added to the input query vector, and the sum of the non-relevant vectors is subtracted in the equation. We use the standard values: alpha = 1, beta = 0.75, gamma = 0.15. The resulting new query vector is then sorted by tf-idf values, and the top two terms are selected to expand the query for next information retrieval. Top terms are compared against the terms already existing in the query so that only terms that do not have the same stem words from the query are chosen.

After the two additional terms are selected to be added to the query we determine the most likely ordering of the augmented query. Using Natural Language Toolkit (NLTK) we tokenize the text gathered from the document titles and snippets from documents marked as relevant using NLTKs built in tokenzier. We then use NLTKs Maxiumum Liklihood Estimator model (MLE) to train on the provided corpus. Using the query and the two additional selected words, the full augmented query is converted to a list and every permutation of the total words is found. Using each possible ordering of terms the total probability of that ordering is calculated by multiplying the probability of each ngram using the trained MLE. The permutation and its probability are stored in a dictionary, which is then sorted is descending order, and the ordering with the highest probability is selected and returned as the final optimized augmented query.

### Example Cases
Detailed transcript of all runs are in the `tests` folder as txt file.

1. Per se restaurant in New York City, starting with the query "per se"
```
$ python3 main.py AIzaSyBzYaejfeOM_u2HXn4BCGgVV0iF6BACQrA 7d381fff3bad9e791 0.9 "per se"
```
The query is augmented to "michelin restaurant per se" in the second iteration and hits precision 1.0.

2. Google cofounder Sergey Brin, starting with the query "brin"
```
$ python3 main.py AIzaSyBzYaejfeOM_u2HXn4BCGgVV0iF6BACQrA 7d381fff3bad9e791 0.9 "brin"
```
The query is augmented to "brin sergey larry" in the second iteration and hits precision 0.9.

3. Covid-19 cases, starting with the query "cases"
```
$ python3 main.py AIzaSyBzYaejfeOM_u2HXn4BCGgVV0iF6BACQrA 7d381fff3bad9e791 0.9 "cases"
```
The query is augmented to "cases data deaths" in the second iteration and hits precision 1.0.


## Notes
- The external library `PorterStemmer` is utilized in comparing candidate terms and query terms to avoid extending query by adding a redundant term. For example, if the term "chocolate" is already present in the query, then the term "chocolatey" would not be considered as the new term while query expansion.
- Non-HTML Files: We are only using the title and snippet fetched by the Google API and not analyzing entire HTML files.
- The MLE used from the NLTK package is trained on the provided text (all the titles and snippets of relevant documents). The liklihood of each 2 terms ocurring in each order is then calculated and used to find the final ordering. For example, if the orginal query is "Taylor Love Story" and the two words selected to be added are "Swift" and "song", the algorithm finds every possible ordering of the terms (i.e Taylor Love Story Swift Song, Taylor Love Swift Song Story, etc.) and multiplies the probability of each bigram appearing in the relevant documents (i.e P(Taylor) * P(Taylor, Swift) * P(Swift, Love) * P(Love, Story) ) to find the total probability of each ordering. The most likely ordering of terms based on the relevant documents is returned with the highest probability.