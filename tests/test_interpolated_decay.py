import tensorflow as tf
from interpolated_learning_rate import interpolated_decay

def almost_equal(x,y):
	threshold=x*1e-7
	return abs(x-y) < threshold

with tf.Session() as sess:
	test_1 = interpolated_decay(1e-3, 1e-7, 10, 100, 200)
	sol_1 = 0.001
	assert almost_equal(sess.run(test_1), sol_1)

	test_2 = interpolated_decay(1e-3, 1e-7, 100, 100, 200)
	sol_2 = 0.001
	assert almost_equal(sess.run(test_2), sol_2)

	test_3 = interpolated_decay(1e-3, 1e-7, 125, 100, 200)
	sol_3 = 0.000750025
	assert almost_equal(sess.run(test_3), sol_3)

	test_4 = interpolated_decay(1e-3, 1e-7, 182, 100, 200)
	sol_4 = 0.000180082
	assert almost_equal(sess.run(test_4), sol_4)

	test_5 = interpolated_decay(1e-3, 1e-7, 200, 100, 200)
	sol_5 = 0.0000001
	assert almost_equal(sess.run(test_5), sol_5)

	test_6 = interpolated_decay(1e-3, 1e-7, 200000, 100, 200)
	sol_6 = 0.0000001
	assert almost_equal(sess.run(test_6), sol_6)

	test_7 = interpolated_decay(1.0, .2, 0, 1e5, 9e5)
	sol_7 = 1.0
	assert almost_equal(sess.run(test_7), sol_7)

	test_8 = interpolated_decay(1.0, .2, 109, 1e5, 9e5)
	sol_8 = 1.0
	assert almost_equal(sess.run(test_8), sol_8)
	
	test_9 = interpolated_decay(1.0, .2, 3e5, 1e5, 9e5)
	sol_9 = 0.8
	assert almost_equal(sess.run(test_9), sol_9)

	test_10 = interpolated_decay(1.0, .2, 5e5, 1e5, 9e5)
	sol_10 = 0.6
	assert almost_equal(sess.run(test_10), sol_10)

	test_11 = interpolated_decay(1.0, .2, 7e5, 1e5, 9e5)
	sol_11 = 0.4
	assert almost_equal(sess.run(test_11), sol_11)

	test_12 = interpolated_decay(1.0, .2, 9e5, 1e5, 9e5)
	sol_12 = 0.2
	assert almost_equal(sess.run(test_12), sol_12)

	test_13 = interpolated_decay(1.0, .2, 9e19, 1e5, 9e5)
	sol_13 = 0.2
	assert almost_equal(sess.run(test_13), sol_13)

	print('\npassed all!\n')
