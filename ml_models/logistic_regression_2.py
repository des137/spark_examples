#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def main():
    spark = SparkSession.builder.appName('logregconsult').getOrCreate()
    data = spark.read.csv('./data/customer_churn.csv', inferSchema=True,
                          header=True)
    data.printSchema()
    data.describe().show()
    data.columns
    assembler = VectorAssembler(inputCols=['Age', 'Total_Purchase',
                                           'Account_Manager', 'Years',
                                           'Num_Sites'],
                                outputCol='features')
    output = assembler.transform(data)
    final_data = output.select('features', 'churn')
    train_churn, test_churn = final_data.randomSplit([0.7, 0.3])
    lr_churn = LogisticRegression(labelCol='churn')
    fitted_churn_model = lr_churn.fit(train_churn)
    training_sum = fitted_churn_model.summary
    training_sum.predictions.describe().show()
    pred_and_labels = fitted_churn_model.evaluate(test_churn)
    pred_and_labels.predictions.show()
    churn_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                               labelCol='churn')
    auc = churn_eval.evaluate(pred_and_labels.predictions)
    print(auc)
    final_lr_model = lr_churn.fit(final_data)
    new_customers = spark.read.csv('new_customers.csv', inferSchema=True,
                                   header=True)
    new_customers.printSchema()
    test_new_customers = assembler.transform(new_customers)
    test_new_customers.printSchema()
    final_results = final_lr_model.transform(test_new_customers)
    final_results.select('Company', 'prediction').show()

if __name__ == '__main__':
    main()
