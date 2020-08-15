import data_service, plot_service, tf_utils

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = data_service.load_dataset()
plot_service.plot_sample(X_train_orig, Y_train_orig)
X_train, Y_train, X_test, Y_test = data_service.preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)

parameters = tf_utils.model(X_train, Y_train, X_test, Y_test)

