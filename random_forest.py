#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (RandomForestClassifier,
                                       DecisionTreeClassifier)


def main():
    spark = SparkSession.builder.appName('dogfood').getOrCreate()
    data = spark.read.csv('./data/dog_food.csv', inferSchema=True, header=True)
    data.printSchema()
    data.head()
    data.describe().show()
    data.columns
    assembler = VectorAssembler(inputCols=['A', 'B', 'C', 'D'],
                                outputCol="features")
    output = assembler.transform(data)
    rfc = DecisionTreeClassifier(labelCol='Spoiled', featuresCol='features')
    output.printSchema()
    final_data = output.select('features', 'Spoiled')
    final_data.head()
    rfc_model = rfc.fit(final_data)
    print(rfc_model.featureImportances)

if __name__ == '__main__':
    main()
