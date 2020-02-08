#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct, avg, stddev
from pyspark.sql.functions import format_number


def main():
    spark = SparkSession.builder.appName('aggs').getOrCreate()
    df = spark.read.csv('./data/sales_info.csv', inferSchema=True, header=True)
    df.show()
    df.printSchema()
    df.groupBy('Company')
    df.groupBy('Company').mean().show()
    df.groupBy('Company').sum().show()
    df.groupBy('Company').max().show()
    df.groupBy('Company').count().show()
    df.agg({'Sales': 'max'}).show()
    group_data = df.groupBy('Company')
    group_data.agg({'Sales': 'max'}).show()
    df.select(countDistinct('Sales')).show()
    df.select(avg('Sales').alias('Average Sales')).show()
    df.select(stddev('Sales')).show()
    sales_std = df.select(stddev('Sales').alias('std'))
    sales_std.select(format_number('std', 2).alias('std')).show()
    df.show()
    df.orderBy('Sales').show()
    df.orderBy(df['Sales'].desc()).show()

if __name__ == '__main__':
    main()
