import tensorflow as tf
w=tf.Variable(tf.constant(5,dtype=tf.float32))

#损失函数 loss=(w+1)^2, w初值设为5，反向传播就是求最优w，是loss最小
loss = tf.square(w+1)

##反向传播算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#生成会话，训练40轮
with tf.compat.v1.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(40):
		sess.run(train_step)
		w_val=sess.run(w)
		loss_val=sess.run(loss)
		print("after %s steps:w is %f, loss is %f" % (i,w_val,loss_val))
