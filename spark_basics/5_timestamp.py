#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import (dayofmonth, hour, dayofyear, month, year,
                                   weekofyear, format_number, date_format)


def main():
    spark = SparkSession.builder.appName('dates').getOrCreate()
    df = spark.read.csv('./data/appl_stock.csv', header=True, inferSchema=True)
    df.head(1)
    df.select(['Date', 'Open']).show()
    df.select(dayofmonth(df['Date'])).show()
    df.select(hour(df['Date'])).show()
    df.select(month(df['Date'])).show()
    df.select(year(df['Date'])).show()
    new_df = df.withColumn('Year', year(df['Date']))
    result = new_df.groupBy('Year').mean().select(['Year', 'avg(Close)'])
    result.select(['Year', format_number('avg(Close)', 2).alias('Avg Close')]).show()

if __name__ == '__main__':
    main()
