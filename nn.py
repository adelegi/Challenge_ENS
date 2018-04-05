import data
import features
import toolkit
import sklearn
import time

from sklearn.neural_network import MLPRegressor

# parameters
reg = 10


temp, dico = data.load_input_data('data/train_input.csv')
output = data.load_output_data('data/challenge_output.csv', temp, dico)
all_features = features.load_all_features(dico, temp, remove_useless=True)

print('features extracted')

for t in range(5):

	train, test = toolkit.random_drawing(all_features, output)

	x_train = toolkit.preprocess(train[0])
	x_test = toolkit.preprocess(test[0])

	for n, model in enumerate([(300, 100, 20), (300, 100), (300)]):

		pred = []
		for i in range(5):

			y_train = train[1][:, i]
			y_test = test[1][:, i]

			neural_net = MLPRegressor(hidden_layer_sizes=model, validation_fraction = 0.2, early_stopping = False, 
                          verbose = True, random_state = 777, learning_rate='constant', alpha = reg, max_iter = 1000,
                          learning_rate_init=0.0001, tol=1e-6)

			start = time.time()
			neural_net.fit(x_train, y_train)
			end = time.time()

			print("Training time: " + str(end-start))

			y_hat = neural_net.predict(x_test)

			mse = sklearn.metrics.mean_squared_error(y_hat, y_test)
			pred.append(mse)

			print('\n {} \n'.format(pred))

		np.savetxt('data/model_{}_val{}.txt'.format(n, t), pred)




