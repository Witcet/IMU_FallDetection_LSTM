import os
import pickle
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import RealDataLoader, SimuDataLoader, ZhjiDataLoader, MixData
import tensorflow as tf

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

TB_DIR = "for_tensorboard"
model_dir = "saved_model"


class FallDetector:
    def __init__(self, batch_size, num_features, timesteps, num_classes):
        self.batch_size = batch_size
        self.num_features = num_features
        self.timesteps = timesteps
        self.num_classes = num_classes

        os.makedirs(model_dir, exist_ok=True)
        self.model_file = os.path.join(model_dir, "model.ckpt")

        print("Resetting the default graph...")
        self.build_model()
        self.define_loss()
        self.define_optimizer()

        self.saver = tf.train.Saver()
        self.tensorboard_op = tf.compat.v1.summary.merge_all()
        print("Model built!")

    def build_model(self):
        # Before beginning, reset the graph.
        tf.compat.v1.reset_default_graph()

        # Define placeholders
        self.x = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, self.timesteps, self.num_features),
                                          name='x')
        self.y = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, self.num_classes), name='y')
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')

        # Normalize input
        normalized_input = tf.layers.batch_normalization(self.x, axis=2, training=self.is_training)

        # Define RNN layers
        rnn_cell = tf.compat.v1.nn.rnn_cell.GRUCell(64)
        rnn_output, _ = tf.compat.v1.nn.dynamic_rnn(rnn_cell, normalized_input, dtype=tf.float32)

        # Flatten the output from RNN layers
        flattened = tf.layers.flatten(rnn_output)

        # Dense layers
        dense1 = tf.layers.dense(inputs=flattened, units=128, activation=tf.nn.relu, name='dense1')
        dense2 = tf.layers.dense(inputs=dense1, units=self.num_classes, activation=tf.nn.softmax, name='dense2')

        self.prediction = dense2

        # Accuracy
        pred_classes = tf.argmax(self.prediction, axis=-1)
        actual_classes = tf.argmax(self.y, axis=-1)
        correct = tf.equal(pred_classes, actual_classes)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    def define_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.prediction))
        tf.summary.scalar("Cross-entropy-loss", self.loss)

    def define_optimizer(self):
        optimizer = tf.train.AdamOptimizer(0.0001)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)

    def load_dataset(self, path, dataloader, use_type):
        if os.path.isfile(f"{use_type}_dataset.pkl"):
            os.remove(f"{use_type}_dataset.pkl")

        loader_map = {"simu": SimuDataLoader,
                      "real": RealDataLoader,
                      "xsen": ZhjiDataLoader}

        if dataloader in loader_map.keys():
            data = loader_map[dataloader](path)
            pickle.dump(data, open(f"{use_type}_dataset.pkl", 'wb'))
        else:
            print("Error! Unknown data loader.")

    def train(self, epochs=50, log_step=10, resume=False):
        print("Beginning the training process...")

        data = pickle.load(open("train_dataset.pkl", 'rb'))

        valid_file = os.path.join(model_dir, "min_valid_loss.txt")
        if os.path.isfile(valid_file):
            min_valid_loss = float(open(valid_file).read().strip())
        else:
            min_valid_loss = 1000000.0

        with tf.Session() as session:
            writer = tf.compat.v1.summary.FileWriter(TB_DIR, session.graph)

            if resume:
                try:
                    self.saver.restore(session, self.model_file)
                    print("Previous checkpoint file found, continue training...")
                except:
                    print("No previous checkpoint file found, restarting training...")

            else:
                min_valid_loss = 1000000.0
                session.run(tf.compat.v1.global_variables_initializer())

            for e in range(epochs):
                avg_loss = []
                for batch_x, batch_y in data.dataset.next_batch(self.batch_size, training=True):
                    batch_loss, _, tb_op = session.run([self.loss, self.train_step, self.tensorboard_op],
                                                       feed_dict={self.x: batch_x,
                                                                  self.y: batch_y,
                                                                  self.is_training: True})
                    avg_loss.append(batch_loss)
                print("Average Loss for epoch {} = {}.".format(e + 1, sum(avg_loss) / len(avg_loss)))

                if (e + 1) % log_step == 0:
                    avg_loss = []
                    avg_accuracy = []
                    all_predictions = []
                    all_labels = []
                    for batch_x, batch_y in data.dataset.next_batch(self.batch_size, validate=True):
                        batch_loss, batch_acc, batch_predictions, batch_labels = session.run([self.loss,
                                                                                              self.accuracy,
                                                                                              self.prediction,
                                                                                              self.y],
                                                                                               feed_dict={self.x: batch_x,
                                                                                                          self.y: batch_y,
                                                                                                          self.is_training: False})
                        avg_loss.append(batch_loss)
                        avg_accuracy.append(batch_acc)
                        all_predictions.extend(batch_predictions)
                        all_labels.extend(batch_labels)

                    try:
                        avg_loss = sum(avg_loss) / len(avg_loss)
                        avg_accuracy = sum(avg_accuracy) / len(avg_accuracy)
                        print("Validation Loss: {}. Validation Accuracy: {}".format(avg_loss, avg_accuracy))
                    except ZeroDivisionError:
                        avg_loss = 1000000.0
                        avg_accuracy = 0
                        print("Error! The accuracy of the model on the validation set is 0.")

                    # Calculate precision, recall, and F1-score
                    all_predictions = np.argmax(all_predictions, axis=-1)
                    all_labels = np.argmax(all_labels, axis=-1)
                    precision = precision_score(all_labels, all_predictions, average='macro')
                    recall = recall_score(all_labels, all_predictions, average='macro')
                    f1 = f1_score(all_labels, all_predictions, average='macro')

                    print("Validation Precision: {}. Validation Recall: {}. Validation F1-score: {}".format(precision, recall, f1))

                    if avg_loss < min_valid_loss:
                        min_valid_loss = avg_loss
                        with open(valid_file, 'w') as f:
                            f.write(str(avg_loss))
                        fname = self.saver.save(session, self.model_file)
                        print("Session saved in {}".format(fname))

                writer.add_summary(tb_op, e)

            writer.close()

        print("Training complete!")

    def test(self):
        print("Beginning the testing process...")

        data = pickle.load(open("test_dataset.pkl", 'rb'))

        with tf.Session() as session:
            self.saver.restore(session, self.model_file)

            avg_loss = []
            avg_accuracy = []
            all_predictions = []
            all_labels = []
            for batch_x, batch_y in data.dataset.next_batch(batch_size=16, test=True):
                batch_loss, batch_acc, batch_predictions, batch_labels = session.run([self.loss,
                                                                                      self.accuracy,
                                                                                      self.prediction,
                                                                                      self.y],
                                                                                       feed_dict={self.x: batch_x,
                                                                                                  self.y: batch_y,
                                                                                                  self.is_training: False})
                avg_loss.append(batch_loss)
                avg_accuracy.append(batch_acc)
                all_predictions.extend(batch_predictions)
                all_labels.extend(batch_labels)

            avg_loss = sum(avg_loss) / len(avg_loss)
            avg_accuracy = sum(avg_accuracy) / len(avg_accuracy)
            print("Test complete!")
            print("Average loss on test set: {}".format(avg_loss))
            print("Average accuracy of test dataset: {}".format(avg_accuracy))

            # Calculate precision, recall, and F1-score
            all_predictions = np.argmax(all_predictions, axis=-1)
            all_labels = np.argmax(all_labels, axis=-1)
            precision = precision_score(all_labels, all_predictions, average='macro')
            recall = recall_score(all_labels, all_predictions, average='macro')
            f1 = f1_score(all_labels, all_predictions, average='macro')

            print("Test Precision: {}. Test Recall: {}. Test F1-score: {}".format(precision, recall, f1))


if __name__ == "__main__":
    fallDetector = FallDetector(16, 12, 240, 2)

    # 如果不需要重新加载训练数据，注释掉下面两行
    # train_path = r"D:\pycharm\PycharmProjects\LSTM\Video_simu_dataset"
    # fallDetector.load_dataset(train_path, dataloader="simu", use_type="train")

    # 如果不需要重新训练，注释掉下面一行
    fallDetector.train(epochs=50, resume=False)

    # 如果不需要重新加载测试数据，注释掉下面两行
    # test_path = r"D:\pycharm\PycharmProjects\LSTM\Thesis_Fall_Dataset"
    # fallDetector.load_dataset(test_path, dataloader="real", use_type="test")

    fallDetector.test()
