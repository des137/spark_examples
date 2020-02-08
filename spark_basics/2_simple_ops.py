#!/usr/bin/env python3
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.appName('ops').getOrCreate()
    df = spark.read.csv('./data/appl_stock.csv', inferSchema=True, header=True)
    df.printSchema()
    df.describe().show()
    df.show()
    df.head(3)[0]
    df.filter('Close < 500').select(['Open', 'Close']).show()
    df.filter(df['Close'] < 500).select('Volume').show()
    df.filter((df['Close'] < 200) & ~(df['Open'] > 200)).show()
    df.filter(df['Low'] == 197.16).show()
    result = df.filter(df['Low'] == 197.16).collect()
    row = result[0]
    row.asDict()

if __name__ == '__main__':
    main()
