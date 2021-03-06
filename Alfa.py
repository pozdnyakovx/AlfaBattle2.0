
# Alfa Battle 2.0 baseline
# Task 2 (default prediction)

# Features:
# * app_id - Идентификатор заявки. заявки пронумерованы так, что более поздним заявкам соответствует # более поздняя дата
# * amnt - Нормированная сумма транзакции. 0.0 - соответствует пропускам 
# * currency - Идентификатор валюты транзакции
# * operation_kind - Идентификатор типа транзакции
# * card_type - Уникальный идентификатор типа карты
# * operation_type - Идентификатор типа операции по пластиковой карте
# * operation_type_group - Идентификатор группы карточных операций, например, дебетовая карта или кредитная карта
# * ecommerce_flag - Признак электронной коммерции
# * payment_system - Идентификатор типа платежной системы
# * income_flag - Признак списания/внесения денежных средств на карту
# * mcc - Уникальный идентификатор типа торговой точки
# * country - Идентификатор страны транзакции
# * city - Идентификатор города транзакции
# * mcc_category - Идентификатор категории магазина транзакции
# * day_of_week - День недели, когда транзакция была совершена
# * hour - Час, когда транзакция была совершена
# * days_before - Количество дней до даты выдачи кредита
# * weekofyear - Номер недели в году, когда транзакция была совершена
# * hour_diff - Количество часов с момента прошлой транзакции для данного клиента
# * transaction_number - Порядковый номер транзакции клиента
 
# Target:
# * flag - Целевая переменная, 1 - факт ухода в дефолт.
 
# URL: https://boosters.pro/championship/alfabattle2_sand/overview
 
# This is an open-source notebook and can be used accordingly.

# 1. Импорт данных через Spark

from pyspark.sql.types import DoubleType, StringType, StructField, IntegerType, LongType, StructType

schema = StructType([
  StructField("app_id", IntegerType(), True),
  StructField("amnt", DoubleType(), True),
  StructField("currency", IntegerType(), True),
  StructField("operation_kind", IntegerType(), True),
  StructField("operation_type", IntegerType(), True),
  StructField("operation_type_group", IntegerType(), True),
  StructField("ecommerce_flag", IntegerType(), True),
  StructField("payment_system", IntegerType(), True),
  StructField("income_flag", IntegerType(), True),
  StructField("mcc", IntegerType(), True),
  StructField("country", IntegerType(), True),
  StructField("city", IntegerType(), True),
  StructField("mcc_category", IntegerType(), True),
  StructField("day_of_week", IntegerType(), True),
  StructField("hour", IntegerType(), True),
  StructField("days_before", IntegerType(), True),
  StructField("weekofyear", IntegerType(), True),
  StructField("hour_diff", LongType(), True),
  StructField("transaction_number", IntegerType(), True),
  StructField("__index_level_0__", LongType(), True)
])

df = spark.read.format("parquet").schema(schema).load("/FileStore/tables/alfa")
cols = df.columns

df = df.dropna()
df = df.drop_duplicates()

print('Размер датасета: %d x %d' % (df.count(), len(cols)))

display(df)


# 2. Feature engineering

df.select('hour_diff').summary().show()

import pyspark.sql.functions as sf

display(df.groupBy(['payment_system']).agg(sf.count('app_id')))

order_sum_features = df.groupBy(['app_id']).agg(sf.mean('amnt').alias('order_sum_mean'),  # средняя сумма транзакции
                                                   sf.max('amnt').alias('order_sum_max'), # макс. сумма транзакции
                                                   sf.min('amnt').alias('order_sum_min'), # мин. сумма транзакции
                                                   sf.expr('percentile_approx(amnt, 0.5)').alias('order_sum_median'),  # медиана транзакции
                                                   )
display(order_sum_features)


# Credits to @aizakharov94 for heads-up

hour_diff_features = df.select(['app_id', 'hour_diff'])
hour_diff_features = hour_diff_features.withColumn('is_0',
                                                (sf.col('hour_diff') == sf.lit(0)).cast(IntegerType()))  # нулевая разница в транзакциях

# далее последовательно инжинирим фичи для разниц 1-3, 1-10, 5-10, 11-100, 100+

hour_diff_features = hour_diff_features.withColumn('is_1_3',
                ((sf.col('hour_diff') >= sf.lit(1)) & (sf.col('hour_diff') <= sf.lit(3))).cast(IntegerType()))
hour_diff_features = hour_diff_features.withColumn('is_5_10',
                ((sf.col('hour_diff') >= sf.lit(5)) & (sf.col('hour_diff') <= sf.lit(10))).cast(IntegerType()))
hour_diff_features = hour_diff_features.withColumn('is_1_10',
                ((sf.col('hour_diff') >= sf.lit(1)) & (sf.col('hour_diff') <= sf.lit(10))).cast(IntegerType()))
hour_diff_features = hour_diff_features.withColumn('is_11_100',
                ((sf.col('hour_diff') >= sf.lit(11)) & (sf.col('hour_diff') <= sf.lit(100))).cast(IntegerType()))
hour_diff_features = hour_diff_features.withColumn('is_100_plus',
                (sf.col('hour_diff') > sf.lit(100)).cast(IntegerType()))

hour_diff_features = hour_diff_features.groupBy(['app_id']).agg(
                                                            sf.mean('is_0').alias('hour_diff_0_mean'),
                                                            sf.mean('is_1_3').alias('hour_diff_1_3_mean'),
                                                            sf.mean('is_1_10').alias('hour_diff_1_10_mean'),
                                                            sf.mean('is_5_10').alias('hour_diff_5_10_mean'),
                                                            sf.mean('is_11_100').alias('hour_diff_11_100_mean'),
                                                            sf.mean('is_100_plus').alias('hour_diff_100_plus_mean'))

df = df.withColumn('is_night_trans', sf.col('hour').isin({0, 1, 2, 3, 23}).cast(IntegerType()))

target = spark.read.format("csv").load("/FileStore/tables/alfabattle2_sand_alfabattle2_train_target.csv")
cols = df.columns

target = target.withColumnRenamed('_c0','app_id')
target = target.withColumnRenamed('_c1','product')
target = target.withColumnRenamed('_c2','flag')
target = target.filter(target.app_id != 'app_id')

target = target.withColumn('app_id', sf.col('app_id').cast(IntegerType()))
target = target.withColumn('product', sf.col('product').cast(IntegerType()))
target = target.withColumn('flag', sf.col('flag').cast(IntegerType()))

target = target.join(order_sum_features, on=['app_id'], how='left')
target = target.join(hour_diff_features, on=['app_id'], how='left')


# 3. Построение бейзлайна

train_df, test_df = target.randomSplit([0.8, 0.2])


cols = train_df.columns
cols = [i for i in cols if i != 'flag']


# Decision Tree model

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline

vector = VectorAssembler(inputCols=cols, outputCol='features')

tree = DecisionTreeClassifier(featuresCol='features', labelCol='flag')
pipe = Pipeline(stages=[vector, tree])


# Define the pipeline model
pipeModel = pipe.fit(train_df)


# Apply to train data
pred_df = pipeModel.transform(train_df)

# Evaluate the model 

display(pred_df.select("features", "label", "prediction", "probability"))


display(pipelineModel.stages[-1], pred_df.drop("prediction", "rawPrediction", "probability"), "ROC")


# LGBM model

# %pip install lightgbm

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(objective='binary', num_leaves=10, learning_rate=0.05, 
                      max_depth=1, n_estimators=50, boosting_type='goss')

lgbm.fit(train_df.drop('flag', axis=1), train_df['flag'])


y_true = test_df['flag'].values
y_pred = lgbm.predict(test_df.drop('flag', axis=1))

from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)


from sklearn.metrics import f1_score
import numpy as np

def lgb_f1_score(y_hat, y_true):
    y_hat = np.round(y_hat
    return 'f1', f1_score(y_true, y_hat)
  
lgb_f1_score(y_pred,y_true)

