import tensorflow as tf
import numpy as np
BATCH_SIZE=8
seed=23455
tf.compat.v1.disable_eager_execution()

rng=np.random.RandomState(seed)
X=rng.rand(32,2)
#打标签
Y =[[int(x0+x1 < 1)] for(x0,x1) in X]
print("X:",X)
print("Y:",Y)

#定义神经网络的输入，参数，输出，定义前向传播过程
x=tf.compat.v1.placeholder(tf.float32,shape=(None,2))
y_=tf.compat.v1.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random.normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random.normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

loss=tf.reduce_mean(tf.square(y-y_))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
with tf.compat.v1.Session() as sess:
	init_op=tf.global_variables_initializer()
	sess.run(init_op)
	print("w1:",sess.run(w1))
	print("w2:",sess.run(w2))
	
	STEPS = 3000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end=start + BATCH_SIZE
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		if i % 500 == 0:
			total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
			print("After %d trainings steps,loss on data is %g" % (i,total_loss))
	print("w1:",sess.run(w1))
	print("w2",sess.run(w2))

