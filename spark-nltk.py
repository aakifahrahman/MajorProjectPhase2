from pyspark import SparkConf
from pyspark import SparkContext

conf = SparkConf()
conf.setMaster('yarn-client')
conf.setAppName('spark-nltk')
sc = SparkContext(conf=conf)

data = sc.textFile('/files/once.txt')

import nltk

words = data.flatMap(lambda x: nltk.word_tokenize(x))
print words.take(10)

pos_word = words.map(lambda x: nltk.pos_tag([x]))
print pos_word.take(5)
