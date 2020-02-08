#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import (format_number, mean, max, min, count, corr,
                                   year, month)


def main():
    spark = SparkSession.builder.appName('walmart_stock').getOrCreate()
    df = spark.read.csv('./data/walmart_stock.csv', header=True,
                        inferSchema=True)
    df.printSchema()
    df.columns
    df.head(5)
    df.describe().show()
    result = df.describe()
    result.select(result['summary'],
                  format_number(result['Open'].cast('float'), 2).alias('Open'),
                  format_number(result['High'].cast('float'), 2).alias('High'),
                  format_number(result['Low'].cast('float'), 2).alias('Low'),
                  format_number(result['Close'].cast('float'), 2).alias('Close'),
                  result['Volume'].cast('int').alias('Volume')).show()
    df.withColumn('HV Ratio', df['High']/df['Volume']).select('HV Ratio').show()
    df.orderBy(df['High'].desc()).head(1)[0][0]
    df.select(mean(df['Close'])).show()
    df.select(max(df['Volume']), min(df['Volume'])).show()
    df.filter(df['Close'] < 60).select(count(df['Close'])).show()
    df.filter('Close < 60').count()
    df.filter(df['Close'] < 60).count()
    df.filter(df['High'] > 80).count()/df.count()*100
    df.select(corr(df['High'], df['Volume'])).show()
    yeardf = df.withColumn('Year', year(df['Date']))
    max_df = yeardf.groupBy('Year').max()
    max_df.select('Year', 'max(High)').show()
    monthdf = df.withColumn('Month', month('Date'))
    monthavgs = monthdf.select(['Month', 'Close']).groupBy('Month').mean()
    monthavgs.select('Month', 'avg(Close)').orderBy('Month').show()

if __name__ == '__main__':
    main()
