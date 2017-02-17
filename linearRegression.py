
import tensorflow as tf


sess = tf.Session()
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
init = tf.global_variables_initializer()

x = tf.placeholder(tf.float32)
model = W * x + b

y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]


sess.run(init)

for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

out_W, out_b, out_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W : %s b: %s loss: %s"%(out_W, out_b, out_loss))