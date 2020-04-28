#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import corr


def main():
    spark = SparkSession.builder.appName('cruise').getOrCreate()
    df = spark.read.csv('./data/cruise_ship_info.csv', inferSchema=True, header=True)
    df.printSchema()
    df.show()
    df.describe().show()
    df.groupBy('Cruise_line').count().show()
    indexer = StringIndexer(inputCol="Cruise_line", outputCol="cruise_cat")
    indexed = indexer.fit(df).transform(df)
    indexed.head(5)
    print(indexed.columns)
    assembler = VectorAssembler(inputCols=['Age', 'Tonnage', 'passengers',
                                           'length', 'cabins',
                                           'passenger_density', 'cruise_cat'],
                                outputCol="features")
    output = assembler.transform(indexed)
    output.select("features", "crew").show()
    final_data = output.select("features", "crew")
    train_data, test_data = final_data.randomSplit([0.7, 0.3])
    lr = LinearRegression(labelCol='crew')
    lrModel = lr.fit(train_data)
    print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,
                                                  lrModel.intercept))
    test_results = lrModel.evaluate(test_data)
    print("RMSE: {}".format(test_results.rootMeanSquaredError))
    print("MSE: {}".format(test_results.meanSquaredError))
    print("R2: {}".format(test_results.r2))
    df.select(corr('crew', 'passengers')).show()
    df.select(corr('crew', 'cabins')).show()

if __name__ == '__main__':
    main()
