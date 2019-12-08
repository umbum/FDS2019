# packed autoencoder.
# source ; https://github.com/ExcelsiorCJH/Hands-On-ML/blob/master/Chap15-Autoencoders/Chap15-Autoencoders.ipynb

from autoencoder.data_load import source_csv_load
from autoencoder.param import *


################
#  data params #
################
csv_file_path = "./data/bs140513_032310_striped.csv"
source_data, answer_data, data_columns = source_csv_load(csv_file_path, "fraud")
train_x, train_y = source_data, source_data

# Mini-batch
def shuffle_batch(features, labels, batch_size):
    rnd_idx = np.random.permutation(len(features))
    n_batches = len(features) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        batch_x, batch_y = features[batch_idx], labels[batch_idx]
        yield batch_x, batch_y


# Train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            batch_x, batch_y = next(shuffle_batch(train_x, train_y, batch_size))
            sess.run(train_op, feed_dict={inputs: batch_x})
        loss_train = reconstruction_loss.eval(feed_dict={inputs: batch_x})
        saver.save(sess, checkpoint_path, global_step=epoch+1)
        print(f'epoch : {epoch}, Train MSE : {loss_train:.5f}')
