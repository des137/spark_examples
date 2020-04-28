#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler


def main():
    spark = SparkSession.builder.appName('hack_find').getOrCreate()
    dataset = spark.read.csv("./data/hack_data.csv", header=True, inferSchema=True)
    print(dataset.head())
    dataset.describe().show()
    print(dataset.columns)
    feat_cols = ['Session_Connection_Time', 'Bytes Transferred',
                 'Kali_Trace_Used', 'Servers_Corrupted', 'Pages_Corrupted',
                 'WPM_Typing_Speed']
    vec_assembler = VectorAssembler(inputCols=feat_cols, outputCol='features')
    final_data = vec_assembler.transform(dataset)
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=False)
    scalerModel = scaler.fit(final_data)
    cluster_final_data = scalerModel.transform(final_data)
    kmeans3 = KMeans(featuresCol='scaledFeatures', k=3)
    kmeans2 = KMeans(featuresCol='scaledFeatures', k=2)
    model_k3 = kmeans3.fit(cluster_final_data)
    model_k2 = kmeans2.fit(cluster_final_data)
    wssse_k3 = model_k3.computeCost(cluster_final_data)
    wssse_k2 = model_k2.computeCost(cluster_final_data)
    print("With K=3")
    print("Within Set Sum of Squared Errors = " + str(wssse_k3))
    print('--'*30)
    print("With K=2")
    print("Within Set Sum of Squared Errors = " + str(wssse_k2))
    for k in range(2, 9):
        kmeans = KMeans(featuresCol='scaledFeatures', k=k)
        model = kmeans.fit(cluster_final_data)
        wssse = model.computeCost(cluster_final_data)
        print("With K={}".format(k))
        print("Within Set Sum of Squared Errors = " + str(wssse))
        print('--'*30)
    model_k3.transform(cluster_final_data).groupBy('prediction').count().show()
    model_k2.transform(cluster_final_data).groupBy('prediction').count().show()

if __name__ == '__main__':
    main()
