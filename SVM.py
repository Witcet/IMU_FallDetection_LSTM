import os
import pickle
import numpy as np
import warnings
from sklearn.svm import SVC
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
        self.model_file = os.path.join(model_dir, "model.pkl")

        print("Resetting the default graph...")
        self.build_model()
        print("Model built!")

    def build_model(self):
        self.model = SVC(kernel='linear')

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

    def reshape_data(self, data):
        num_samples = data.shape[0]
        reshaped_data = data.reshape(num_samples, -1)
        return reshaped_data

    def preprocess_labels(self, labels):
        return np.argmax(labels, axis=1)

    def train(self):
        print("Beginning the training process...")

        data = pickle.load(open("train_dataset.pkl", 'rb'))

        features = self.reshape_data(data.dataset.features)
        labels = self.preprocess_labels(data.dataset.labels)

        self.model.fit(features, labels)

        pickle.dump(self.model, open(self.model_file, 'wb'))

        print("Training complete!")

    def test(self):
        print("Beginning the testing process...")

        data = pickle.load(open("test_dataset.pkl", 'rb'))

        features = self.reshape_data(data.dataset.features)
        labels = self.preprocess_labels(data.dataset.labels)

        model = pickle.load(open(self.model_file, 'rb'))

        predictions = model.predict(features)
        accuracy = np.mean(predictions == labels)

        print("Test complete!")
        print("Accuracy on test set: {}".format(accuracy))

        # Calculate precision, recall, and F1-score
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')

        print("Test Precision: {}. Test Recall: {}. Test F1-score: {}".format(precision, recall, f1))


if __name__ == "__main__":
    fallDetector = FallDetector(16, 12, 240, 2)

    # 如果不需要重新加载训练数据，注释掉下面两行
    # train_path = r"D:\pycharm\PycharmProjects\LSTM\Video_simu_dataset"
    # fallDetector.load_dataset(train_path, dataloader="simu", use_type="train")

    # 如果不需要重新训练，注释掉下面一行
    fallDetector.train()

    # 如果不需要重新加载测试数据，注释掉下面两行
    # test_path = r"D:\pycharm\PycharmProjects\LSTM\Thesis_Fall_Dataset"
    # fallDetector.load_dataset(test_path, dataloader="real", use_type="test")

    fallDetector.test()