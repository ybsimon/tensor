import numpy as np
import tensorflow as tf

features = [tf.contrib.layers.real_valued_column("", dimension = 1)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns = features)

dataSet = tf.contrib.learn.datasets.base.Dataset(
    data = np.array([[1], [2], [3], [4]]),
    target= np.array([[0], [-1], [-2], [-3]])
)

estimator.fit(x=dataSet.data, y=dataSet.target, steps=1000)

estimator.evaluate(x=dataSet.data, y=dataSet.target)