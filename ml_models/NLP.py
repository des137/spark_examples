#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from pyspark.ml.feature import (Tokenizer, StopWordsRemover, CountVectorizer,
                                IDF, StringIndexer)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def main():
    spark = SparkSession.builder.appName('nlp').getOrCreate()
    data = spark.read.csv("./data/smsspamcollection/SMSSpamCollection",
                          inferSchema=True, sep='\t')
    data = data.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1',
                                                                    'text')
    data.show()
    data = data.withColumn('length', length(data['text']))
    data.show()
    data.groupby('class').mean().show()
    tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
    stopremove = StopWordsRemover(inputCol='token_text',
                                  outputCol='stop_tokens')
    count_vec = CountVectorizer(inputCol='stop_tokens', outputCol='c_vec')
    idf = IDF(inputCol="c_vec", outputCol="tf_idf")
    ham_spam_to_num = StringIndexer(inputCol='class', outputCol='label')
    clean_up = VectorAssembler(inputCols=['tf_idf', 'length'],
                               outputCol='features')
    nb = NaiveBayes()
    data_prep_pipe = Pipeline(stages=[ham_spam_to_num, tokenizer, stopremove,
                                      count_vec, idf, clean_up])
    cleaner = data_prep_pipe.fit(data)
    clean_data = cleaner.transform(data)
    clean_data = clean_data.select(['label', 'features'])
    clean_data.show()
    (training, testing) = clean_data.randomSplit([0.7, 0.3])
    spam_predictor = nb.fit(training)
    data.printSchema()
    test_results = spam_predictor.transform(testing)
    test_results.show()
    acc_eval = MulticlassClassificationEvaluator()
    acc = acc_eval.evaluate(test_results)
    print("Accuracy of model at predicting spam was: {}".format(acc))

if __name__ == '__main__':
    main()
