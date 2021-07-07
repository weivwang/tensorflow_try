搭建模块化的神经网路八股

前向传播就是搭建网络，设计网络结构（forward.py)

``` python
def forward(x,regularizer):
	w=
	b=
	y=
	return y

def get_weigth(shape,regularizer):
  w=tf.Variable()
  #把每一个w的正则化损失加到总损失losses中
  tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
  retrun w
  
#b的形状=某层中b的个数
def get_bias(shape):
  b=tf.Variable()
  return b
```

反向传播就是训练网络，优化网络参数(backward.py)

```python
def backward():
  x=tf.placeholder(  )
  y_=tf.placeholder(  ) #注意下划线
  y=forward.forward(x,REGULARIZER)
  global_step=tf.Variable(0,trainable=False)
  #损失函数
  loss=
  '''
  loss可以是：
  y与y_的差距(loss_mse) = tf.reduce_mean(tf.square(y-y_)
  ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
  cem = tf.reduce_mean(ce)
  加入正则化后：
  loss=y与y_的差距 + tf.add_n(tf.get_collection("losses"))
  '''
  #使用指数衰减学习率，用以下代码：
  learning rate = tf.train.exponential_decay(
  	LEARNING_RATE_BASE,
  	global_step,
  	数据集总样本数/BATCH_SIZE,
  	LEARNING_RATE_DECAY,
  	staircase=True
  	)
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
  
  #滑动平均
  ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
  ema_op = ema.apply(tf.trainable_variables())
  with tf.control_dependencies([train_step,ema_op]):
  	train_op = tf.no_op(name = 'train')
    
  with tf.compat.v1.Session as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    
    for i in range(STEPS):
      sess.run(train_step,feed_dict={x: ,y_: })
      if i % 轮数 == 0:
        print()
   
if __name__ == '__main__':
  backward()

```

