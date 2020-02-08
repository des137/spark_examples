#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean


def main():
    spark = SparkSession.builder.appName('miss').getOrCreate()
    df = spark.read.csv('./data/ContainsNull.csv', header=True,
                        inferSchema=True)
    df.show()
    df.na.drop().show()
    df.na.drop(thresh=2).show()
    df.na.drop(how='all').show()
    df.na.drop(subset=['Sales']).show()
    df.na.fill('Fill Values').show()
    df.na.fill(0).show()
    df.na.fill('No Name', subset=['Name']).show()
    mean_val = df.select(mean(df['Sales'])).collect()
    mean_sales = mean_val[0][0]
    df.na.fill(mean_sales, subset=['Sales']).show()
    df.na.fill(df.select(mean(df['Sales'])).collect()[0][0], ['Sales']).show()

if __name__ == '__main__':
    main()
