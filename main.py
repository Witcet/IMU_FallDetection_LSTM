import os
import pickle
import numpy as np
import warnings
from utils import RealDataLoader, SimuDataLoader, ZhjiDataLoader, MixData

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

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
        self.tensorboard_op = tf.summary.merge_all()
        print("Model built!")

    def build_model(self):
        # Before beginning, reset the graph.
        tf.compat.v1.reset_default_graph()

        # define placeholders
        self.x = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, self.timesteps, self.num_features),
                                          name='x')
        self.y = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, self.num_classes), name='y')
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')

        # normalize input
        normalized_input = tf.layers.batch_normalization(self.x, axis=2, training=self.is_training)

        # convert input tensor to list of tensors
        inputs = tf.split(normalized_input, self.timesteps, axis=1)
        inputs = [tf.squeeze(inp) for inp in inputs]

        # RNN layers
        rnn1_out = self._lstm_layer(inputs, 32, "lstm-1")
        rnn2_out = self._lstm_layer(rnn1_out, 64, "lstm-2")

        # Get last node
        final_node = rnn2_out[-1]

        # Dense layers
        dense1_out = self._dense_layer(final_node, 32, "dense-1", activation=tf.nn.elu)
        dense2_out = self._dense_layer(dense1_out, self.num_classes, "dense-2", activation=tf.nn.softmax)

        self.prediction = dense2_out

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

    def _lstm_layer(self, inputs, num_units, scope, cell_type="LSTM"):
        with tf.variable_scope(scope):
            if cell_type == "LSTM":
                lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
            elif cell_type == "GRU":
                lstm_cell = tf.nn.rnn_cell.GRUCell(num_units)
            else:
                raise NotImplementedError("{} type not implemented.".format(cell_type))

            outputs, states = tf.nn.static_rnn(lstm_cell, inputs, dtype=tf.float32)
        return outputs

    def _dense_layer(self, inputs, num_units, scope, activation=None):
        with tf.variable_scope(scope):
            w = tf.get_variable("w", shape=(inputs.shape[-1].value, num_units))
            b = tf.get_variable("b", shape=(num_units), initializer=tf.constant_initializer(0.1))

            out = tf.matmul(inputs, w) + b
            if activation is not None:
                out = activation(out)

        return out

    def load_dataset(self, path, dataloader, use_type):
        if os.path.isfile(f"{use_type}_dataset.pkl"):
            os.remove(f"{use_type}_dataset.pkl")

        loader_map = {"simu": SimuDataLoader,
                      "real": RealDataLoader,
                      "zhji": ZhjiDataLoader}

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
            writer = tf.summary.FileWriter(TB_DIR, session.graph)

            if resume:
                try:
                    self.saver.restore(session, self.model_file)
                    print("Previous checkpoint file found,continue training...")
                except:
                    print("No previous checkpoint file found,restarting training...")

            else:
                min_valid_loss = 1000000.0
                session.run(tf.global_variables_initializer())

            for e in range(epochs):
                avg_loss = []
                for batch_x, batch_y in data.dataset.next_batch(self.batch_size, training=True):
                    batch_loss, _, tb_op = session.run([self.loss,
                                                        self.train_step,
                                                        self.tensorboard_op],
                                                       feed_dict={self.x: batch_x,
                                                                  self.y: batch_y,
                                                                  self.is_training: True})
                    avg_loss.append(batch_loss)
                print("Average Loss for epoch {} = {}.".format(e + 1, sum(avg_loss) / len(avg_loss)))

                if (e + 1) % log_step == 0:
                    avg_loss = []
                    avg_accuracy = []
                    for batch_x, batch_y in data.dataset.next_batch(self.batch_size, validate=True):
                        batch_loss, batch_acc = session.run([self.loss,
                                                             self.accuracy],
                                                            feed_dict={self.x: batch_x,
                                                                       self.y: batch_y,
                                                                       self.is_training: False})
                        avg_loss.append(batch_loss)
                        avg_accuracy.append(batch_acc)

                    try:
                        avg_loss = sum(avg_loss) / len(avg_loss)
                        avg_accuracy = sum(avg_accuracy) / len(avg_accuracy)
                        print("Validation Loss: {}. Validation Accuracy: {}".format(avg_loss, avg_accuracy))
                    except ZeroDivisionError:
                        avg_loss = 1000000.0
                        avg_accuracy = 0
                        print("Error! The accuracy of the model on the validation set is 0.")

                    if avg_loss < min_valid_loss:
                        min_valid_loss = avg_loss
                        with open(valid_file, 'w') as f:
                            f.write(str(avg_loss))
                        fname = self.saver.save(session, self.model_file)
                        print("Session saved in {}".format(fname))

                writer.add_summary(tb_op, e)

            writer.close()

        print("Training complete!")

        # self.evaluate(data)

    # def evaluate(self, data):   # evaluate方法弃用，改用test方法
    #     with tf.Session() as session:
    #         self.saver.restore(session, self.model_file)
    #
    #         avg_loss = []
    #         avg_accuracy = []
    #         for batch_x, batch_y in data.next_batch(batch_size=16, evaluate=True):
    #             pred, loss_, acc_ = session.run([self.prediction,
    #                                              self.loss,
    #                                              self.accuracy],
    #                                             feed_dict={self.x: batch_x,
    #                                                        self.y: batch_y,
    #                                                        self.is_training: False})
    #             avg_loss.append(loss_)
    #             avg_accuracy.append(acc_)
    #
    #         avg_loss = sum(avg_loss) / len(avg_accuracy)
    #         print("Average loss on evaluation set: {}".format(avg_loss))
    #         print("Average accuracy of evaluate set: {}".format(avg_accuracy))
    #
    #     return avg_loss, avg_accuracy

    def test(self):
        print("Beginning the testing process...")

        data = pickle.load(open("test_dataset.pkl", 'rb'))

        with tf.Session() as session:
            self.saver.restore(session, self.model_file)

            avg_loss = []
            avg_accuracy = []
            TP = 0
            TN = 0
            FN = 0
            FP = 0
            for batch_x, batch_y in data.dataset.next_batch(batch_size=16, test=True):
                pred, loss_, acc_ = session.run([self.prediction,
                                                 self.loss,
                                                 self.accuracy],
                                                feed_dict={
                                                    self.x: batch_x,
                                                    self.y: batch_y,
                                                    self.is_training: False
                                                })
                avg_loss.append(loss_)
                avg_accuracy.append(acc_)
                # Fall:01 ADL:10
                for batch in range(16):  # batch_size=16
                    if (batch_y[batch][0] == 0):  # sample = Fall
                        if (pred[batch][0] < pred[batch][1]):  # pred = Fall
                            TP += 1
                        else:
                            FN += 1
                    else:  # sample = not Fall
                        if (pred[batch][0] < pred[batch][1]):  # pred = Fall
                            FP += 1
                        else:
                            TN += 1

            avg_loss = sum(avg_loss) / len(avg_loss)
            avg_accuracy = sum(avg_accuracy) / len(avg_accuracy)

            precision=TP/(TP+FP)
            recall = TP / ( TP + FN )
            F1 = ( 2* precision * recall ) / (precision + recall)

            print("Test complete!")
            print("Average loss on test set: {}".format(avg_loss))
            print("Average accuracy of test dataset: {}".format(avg_accuracy))
            print('precision:', precision)
            print('recall:', recall)
            print('F1:', F1)
            print('TN', TN)
            print('FP', FP)
            print('ADL_error',FP/(TN+FP))
            print('TP', TP)
            print('FN', FN)
            print('FALL_error',FN/(TP+FN))
            print()
            print(avg_accuracy)
            print(precision)
            print(recall)
            print(F1)
            print(TN)
            print(FP)
            print(FP/(TN+FP))
            print(TP)
            print(FN)
            print(FN/(TP+FN))




if __name__ == "__main__":
    fallDetector = FallDetector(16, 3, 240, 2)

    # 如果不需要重新加载训练数据，注释掉下面两行
    # train_path=r"D:\OpenSim\Datasetall\Video_simu_dataset"
    # train_path = r"D:\OpenSim\Datasetall\zzhangjiangdata\sub6mot29_simu"
    train_path=r"D:\OpenSim\Datasetall\zzhangjiangdata\simu_imu_thesis_sub10"
    fallDetector.load_dataset(train_path,dataloader="simu",use_type="train")

    # 如果不需要重新训练，注释掉下面一行
    fallDetector.train(epochs=50, resume=False)

    # 如果不需要重新加载测试数据，注释掉下面两行l
    test_path = r"D:\OpenSim\Datasetall\Thesis_Fall_Dataset"
    # test_path=r"D:\OpenSim\Datasetall\zzhangjiangdata\real_for_test"
    fallDetector.load_dataset(test_path,dataloader="real",use_type="test")

    fallDetector.test()

# # 真实数据与仿真数据混合，训练模型
# if __name__=="__main__":
#
#     fallDetector = FallDetector(16, 3, 240, 2)
#
#     simu_path=r"D:\OpenSim\Datasetall\zzhangjiangdata\sub6mot29_simu"
#     zhji_path=r"D:\OpenSim\Datasetall\zzhangjiangdata\real_for_train_jmx"
#
#     simu_data=SimuDataLoader(simu_path)
#     zhji_data=ZhjiDataLoader(zhji_path)
#     print(simu_data.dataset.labels.shape[0])
#     print(zhji_data.dataset.labels.shape[0])
#
#     simu_feature=simu_data.dataset.features
#     simu_label=simu_data.dataset.labels
#     zhji_feature=zhji_data.dataset.features
#     zhji_label=zhji_data.dataset.labels
#
#     features=np.concatenate((simu_feature,zhji_feature),axis=0)
#     labels=np.concatenate((simu_label,zhji_label),axis=0)
#
#     mix_data=MixData(features,labels)
#
#     pickle.dump(mix_data,open("train_dataset.pkl",'wb'))
#
#     fallDetector.train(epochs=50)
#
#     test_path = r"D:\OpenSim\Datasetall\zzhangjiangdata\real_for_test"
#     fallDetector.load_dataset(test_path,dataloader="zhji",use_type="test")
#     fallDetector.test()
