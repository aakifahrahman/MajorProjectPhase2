import string
import json 

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
 
# Module-level global variables for the `tokenize` function below
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Function to break text into "tokens", lowercase them, remove punctuation and stopwords, and stem them
def tokenize(text):
    tokens = word_tokenize(text)
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    stemmed = [STEMMER.stem(w) for w in no_stopwords]
    return [w for w in stemmed if w]

# Function assigning class 0 (negative) to reviews with rating 1.0 or 2.0, class 1 (neutral) to reviews with rating 3.0 and class 2 (positive) to reviews with rating 4.0 or 5.0
def class_mapper(label):
    if label == 1.0:
        label_new = 0
    elif label == 2.0:
        label_new = 0
    elif label == 3.0:
        label_new = 0
    elif label == 4.0:
        label_new = 1
    elif label == 5.0:
        label_new = 1
    return label_new

# Initialize a SparkContext
sc = SparkContext()

# Enter a book id
asin = '0002007770'

# Import full dataset of newsgroup posts as text file
data_raw = sc.textFile('/home/theertha/Documents/scala/spark-1.6.0/files/dataset.json')

# Parse JSON entries in dataset
data = data_raw.map(lambda line: json.loads(line))

# Filter those lines that do not have this id, to create training set.
data_train = data.filter(lambda line: line['asin'] != asin)

# Filter only those lines that have this id, to create test set
data_test = data.filter(lambda line: line['asin'] == asin)

# Extract relevant fields in training dataset -- category label and text content
data_relevant_train = data_train.map(lambda line: (line['overall'], line['reviewText']))

# Extract relevant fields in testing dataset -- category label and text content
data_relevant_test = data_test.map(lambda line: (line['overall'], line['reviewText']))

# Prepare text for analysis using our tokenize function to clean it up, on training set
data_cleaned_train = data_relevant_train.map(lambda (label, text): (class_mapper(label), tokenize(text)))

# Prepare text for analysis using our tokenize function to clean it up, on testing set
data_cleaned_test = data_relevant_test.map(lambda (label, text): (class_mapper(label), tokenize(text)))

# Hashing term frequency vectorizer with 50k features, for training set
htf_train = HashingTF(50000)

# Hashing term frequency vectorizer with 50k features, for testing set
htf_test = HashingTF(50000)

# Create an RDD of LabeledPoints using category labels as labels and tokenized, hashed text as feature vectors, for training data
data_hashed_train = data_cleaned_train.map(lambda (label, text): LabeledPoint(label, htf_train.transform(text)))

# Create an RDD of LabeledPoints using category labels as labels and tokenized, hashed text as feature vectors, for testing data
data_hashed_test = data_cleaned_test.map(lambda (label, text): LabeledPoint(label, htf_test.transform(text)))

# Ask Spark to persist the training RDD so it won't have to be re-created later
data_hashed_train.persist()

# Ask Spark to persist the testing RDD so it won't have to be re-created later
data_hashed_test.persist()

# Train a Naive Bayes model on the training data
model = NaiveBayes.train(data_hashed_train)

# Compare predicted labels to actual labels
prediction_and_labels = data_hashed_test.map(lambda point: (model.predict(point.features), point.label))

#Positive Reviews
positive = prediction_and_labels.filter(lambda (predicted, actual): predicted == 1)

#Negative Reviews
negative = prediction_and_labels.filter(lambda (predicted, actual): predicted == 0)

p = positive.count()

n = negative.count()

# Filter to only correct predictions
correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)

#To get the number of labels that are correctly predicted as positive
true_positive = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual and predicted == 1)

tp = true_positive.count()

#To get the number of labels that are wrongly predicted as positive
false_positive = prediction_and_labels.filter(lambda (predicted, actual): predicted != actual and predicted == 1)

fp = false_positive.count()

#To get the number of labels that are wrongly predicted as negative
false_negative = prediction_and_labels.filter(lambda (predicted, actual): predicted != actual and predicted == 0)

fn = false_negative.count()

precision = (float(tp)/float((tp+fp)))
recall = (float(tp)/float((tp+fn)))

# Calculate and print accuracy rate
accuracy = correct.count() / float(data_hashed_test.count())

if (p>n):
	print "\n\n		***The book is POSITIVE and the classifier correctly predicted category " + str(accuracy * 100) + " percent of the time, with precision:" + str(precision*100) + " and recall: " + str(recall*100) + "***	       \n\n"
else: 
	print "\n\n		***The book is NEGATIVE and the classifier correctly predicted category " + str(accuracy * 100) + " percent of the time, with precision:" + str(precision*100) + " and recall: " + str(recall*100) + "***	       \n\n"

