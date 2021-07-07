#两层简单随机网络
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#定义输入和参数
x=tf.constant([[0.7,0.5]])
w1=tf.Variable(tf.random.normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random.normal([3,1],stddev=1,seed=1))

#定义前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
with tf.compat.v1.Session() as sess:
	init_op=tf.global_variables_initializer()
	sess.run(init_op)
	print("y in tf3_2.py is:")
	print(sess.run(y))

