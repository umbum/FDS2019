# packed autoencoder.
# source ; https://github.com/ExcelsiorCJH/Hands-On-ML/blob/master/Chap15-Autoencoders/Chap15-Autoencoders.ipynb

from autoencoder.data_load import source_csv_load
from autoencoder.param import *

################
#  data params #
################
base_path = "../data"
base_csv_file = "bs140513_032310_striped"
from_csv_file_path = os.path.join(base_path, f"{base_csv_file}.csv")
to_csv_file_path = os.path.join(base_path, f"{base_csv_file}_autoencoded.csv")
hashed_csv_file_path = os.path.join(base_path, f"{base_csv_file}_input.csv")
label_csv_file_path = os.path.join(base_path, f"{base_csv_file}_label.csv")

source_data, _, data_columns = source_csv_load(from_csv_file_path, "fraud")
train_x, train_y = source_data, source_data

# load session
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(SAVER_DIR))
    results = sess.run(outputs, feed_dict={inputs:source_data})

data = pd.DataFrame(results, columns=data_columns).set_index('step')
source_data, answer_data, _ = source_csv_load(from_csv_file_path, "fraud", as_numpy=False)
source_data = source_data.set_index('step')

data.to_csv(to_csv_file_path)
answer_data.to_frame('label').set_index('label').to_csv(label_csv_file_path)
source_data.to_csv(hashed_csv_file_path)

