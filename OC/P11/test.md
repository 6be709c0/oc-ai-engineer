In this jupyter notebook code

```python
!pip install Pandas pillow tensorflow pyspark pyarrow
import pandas as pd
from PIL import Image
import numpy as np
import io
import os

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, element_at, split
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
import matplotlib.pyplot as plt
import pandas as pd
PATH = os.getcwd()
PATH_Data = PATH+'/input/Test1'
PATH_Result = PATH+'/output/Results'
print('PATH:        '+\
      PATH+'\nPATH_Data:   '+\
      PATH_Data+'\nPATH_Result: '+PATH_Result)
spark = (SparkSession
             .builder
             .appName('P11')
             .master('local')
             .config("spark.sql.parquet.writeLegacyFormat", 'true')
             .getOrCreate()
)
sc = spark.sparkContext
spark
def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)

def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
    '''
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)
def model_fn():
    """
    Returns a MobileNetV2 model with top layer removed 
    and broadcasted pretrained weights.
    """
    model = MobileNetV2(weights='imagenet',
                        include_top=True,
                        input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    new_model = Model(inputs=model.input,
                  outputs=model.layers[-2].output)
    new_model.set_weights(brodcast_weights.value)
    return new_model
model = MobileNetV2(weights='imagenet',
                    include_top=True,
                    input_shape=(224, 224, 3))

new_model = Model(inputs=model.input,
                  outputs=model.layers[-2].output)

new_model.summary()

brodcast_weights = sc.broadcast(new_model.get_weights())
images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load(PATH_Data)
print(images.count())
images = images.limit(300)
print(images.count())
images = images.withColumn('label', element_at(split(images['path'], '/'),-2))
print(images.printSchema())
print(images.select('path','label').show(5,False))
features_df = images.repartition(20).select(col("path"),
                                            col("label"),
                                            featurize_udf("content").alias("features")
                                           )

array_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
features_df = features_df.withColumn("features_vector", array_to_vector_udf(col("features")))

# Standardize the data using StandardScaler
scaler = StandardScaler(inputCol="features_vector", outputCol="features_scaled", withStd=True, withMean=True)
scaler_model = scaler.fit(features_df)
features_df = scaler_model.transform(features_df)
num_features = len(features_df.select("features").first()[0])
print(f"Number of features: {num_features}")

rows_count = features_df.count()
print(f"Shape of features_df: ({rows_count}, {num_features + 1})")
pca = PCA(k=num_features, inputCol="features_scaled", outputCol="pca_features")
pca_model = pca.fit(features_df)

# Calculer la variance expliquée cumulée
explained_variance_ratio = pca_model.explainedVariance.cumsum()  # Variance expliquée cumulée
components = np.arange(len(explained_variance_ratio)) + 1

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(components, explained_variance_ratio, label='Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance (%)')
plt.title('PCA Explained Variance')

# Annoter des pourcentages spécifiques de variance
variance_thresholds = [0.9, 0.95, 0.97, 0.98, 0.99]
for threshold in variance_thresholds:
    indices = np.where(explained_variance_ratio >= threshold)[0]
    if indices.size > 0:
        component_number = indices[0]
        plt.scatter(component_number + 1, explained_variance_ratio[component_number], color='red')
        plt.annotate(f"{int(threshold*100)}%", 
                     (component_number + 1, explained_variance_ratio[component_number]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

plt.grid(True)
plt.show()
num_components_95_variance = np.where(explained_variance_ratio >= 0.95)[0][0] + 1
print(f"Number of components for 95% variance: {num_components_95_variance}")
pca_95 = PCA(k=num_components_95_variance, inputCol="features_scaled", outputCol="pca_features_95")
pca_model_95 = pca_95.fit(features_df)
features_df_pca_95 = pca_model_95.transform(features_df)
# Save the PCA-transformed features
features_df_pca_95.write.mode("overwrite").parquet(PATH_Result + "_pca_95")
features_df_pca_95.select("features").show(5)
features_df_pca_95.select("features_scaled").show(5)
features_df_pca_95.select("pca_features_95").show(5)
features_df_pca_95.show(5)
final_result = features_df_pca_95.select('path','label','pca_features_95')
final_result.toPandas().to_csv('./output/final_result.csv')
df_pca = spark.read.parquet(PATH_Result + "_pca_95", engine='pyarrow')
first_row = df_pca.first()
first_row
print("Avant", len(first_row["features"]))
print("Après", len(first_row["pca_features_95"]))
```

Is it possible to display the first 5 images using the "features" columns that is an array 1280 values ?
