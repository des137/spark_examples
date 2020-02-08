#!/usr/bin/env python3
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType


def main():
    spark = SparkSession.builder.appName('Basics').getOrCreate()
    df = spark.read.json('people.json')
    df.show()
    df.printSchema()
    print(df.columns)
    df.describe().show()
    data_schema = [StructField('age', IntegerType(), True),
                   StructField('name', StringType(), True)]
    final_struc = StructType(fields=data_schema)
    df = spark.read.json('people.json', schema=final_struc)
    df.printSchema()
    type(df['age'])
    df.select('age').show()
    df.select(['age', 'name']).show()
    df.withColumn('double_age', df['age']*2).show()
    df.show()
    df.withColumnRenamed('age', 'new_age').show()
    df.createOrReplaceTempView('people')
    results = spark.sql('SELECT * FROM people')
    results.show()
    new_results = spark.sql('SELECT * FROM people WHERE age=30')
    new_results.show()

if __name__ == '__main__':
    main()
