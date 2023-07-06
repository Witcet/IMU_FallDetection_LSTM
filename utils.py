import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal

TIME_STEPS = 180


class Data:
    def __init__(self, features, labels, onehot=True):
        self.features = features
        self.labels = labels

        # convert labels to one-hot
        if onehot:
            num_classes = len(np.unique(self.labels))
            temp = np.zeros((self.labels.shape[0], num_classes))
            temp[range(self.labels.shape[0]), self.labels] = 1  # Fall:01 ADL:10
            self.labels = temp

        # shuffle data
        self.shuffle()

    def shuffle(self):
        np.random.seed(7)
        indices = np.array(range(self.labels.shape[0]))  # WHY : self.labels.shape
        np.random.shuffle(indices)
        self.features = self.features[indices]
        self.labels = self.labels[indices]

    def next_batch(self, batch_size=16, training=False, validation=False, evaluate=False, test=False):
        features = self.features
        labels = self.labels

        train_size = int(features.shape[0] * 0.8)
        valid_size = int(features.shape[0] * 0.9)

        if test:
            pass
        elif training:
            features = features[:train_size]
            labels = labels[:train_size]
        elif validation:
            features = features[train_size:valid_size]
            labels = labels[train_size:valid_size]
        elif evaluate:
            features = features[valid_size:]
            labels = labels[valid_size:]
        else:
            print("Error! Specify one mode!")

        num_samples = features.shape[0]
        n_batches = int(np.ceil(num_samples / batch_size))

        for b in tqdm(range(n_batches), file=sys.stdout):
            start = b * batch_size
            end = start + batch_size
            if end >= num_samples:
                continue
            feature_batch = features[start:end]
            label_batch = labels[start:end]

            yield feature_batch, label_batch


class SimuDataLoader:
    def __init__(self, path=""):
        self.path = path
        self.UseMot = ["801", "802", "803", "804", "805", "806", "807", "808", "809", "810", "811", "812",
                       "901", "902", "903", "904", "907", "908", "909", "910", "911", "912", "913", "915"]
        self.dataset = {}

        self._load_data()
        self._format_dataset()

    def _load_data(self):
        print("Loading dataset...")
        subdirs = list(filter(os.path.isdir, [os.path.join(self.path, sd) for sd in os.listdir(self.path)]))

        for subject in subdirs:
            print(subject)
            trials = [os.path.join(subject, file) for file in os.listdir(subject)]
            filter_trials = []

            for i in range(len(trials)):
                filename = os.path.basename(trials[i])
                mot = filename[0:3]
                if mot in self.UseMot:
                    filter_trials.append(trials[i])

            for i in tqdm(range(len(filter_trials)), file=sys.stdout):
                h5_file = filter_trials[i]
                mot_type = self._adl_or_fall(h5_file)
                df = pd.read_hdf(h5_file, key='data')

                # ARG select imus on which body segment
                # segments= ["torso","pelvis","femur_r","tibia_r"]
                segments = ["tibia_r"]

                each_data = [[]]
                for segment in segments:
                    # ARG select imu data on which imu axis
                    # feature = np.array([df[segment + "_Accel_X"],
                    #                     df[segment + "_Accel_Y"],
                    #                     df[segment + "_Accel_Z"],
                    #                     df[segment + "_Gyro_X"],
                    #                     df[segment + "_Gyro_Y"],
                    #                     df[segment + "_Gyro_Z"]])
                    feature = np.array([df[segment + "_Accel_X"],
                                        df[segment + "_Accel_Y"],
                                        df[segment + "_Accel_Z"]])

                    feature = feature.T
                    interpolate_1d = lambda col: signal.resample(col, TIME_STEPS)
                    feature_new = np.apply_along_axis(interpolate_1d, 0, feature)
                    try:
                        each_data = np.concatenate((each_data, feature_new), axis=1)
                    except ValueError:
                        each_data = feature_new
                each_data = np.expand_dims(each_data, axis=0)

                try:
                    self.dataset[mot_type] = np.concatenate((self.dataset[mot_type], each_data), axis=0)
                except KeyError:
                    self.dataset[mot_type] = each_data

    def _format_dataset(self):
        print("Formatting dataset...")
        readings = None
        labels = []
        for key in self.dataset:
            if key == "FALL":
                mot_type_num = 1
            else:
                mot_type_num = 0
            features = self.dataset[key]
            num_samples = features.shape[0]
            labels.extend(num_samples * [mot_type_num])
            try:
                readings = np.concatenate((readings, features), axis=0)
            except ValueError:
                readings = features
        labels = np.array(labels)
        self.dataset = Data(readings, labels)

    def _adl_or_fall(self, h5file):
        types = {"8": "ADL",
                 "9": "FALL"}
        prefix_1 = os.path.basename(h5file)[0]
        mot_type = types.get(prefix_1)
        return mot_type


class RealDataLoader:
    def __init__(self, path=""):
        self.path = path
        self.ADL_TYPES = ["801-Yurume Ileri", "802-YurumeGeri", "803-TempoluYavasKosu", "804-ComelmeKalkma",
                          "805-Egilme", "806-EgilmeVeAlma", "807-Topallama", "808-Tokezleme",
                          "809-AyakBurkulmasi", "810-Oksurme", "811-Sandalye", "812-Kanepe"]
        self.FALL_TYPES = ["901-OneDogruUzanma", "902-OneDogruKorunarakDusme", "903-OneDogruDizlerUzerine",
                           "904-OneDogruDizlerUzerineArdindanUzanma", "907-OneDogruSagli", "908-OneDogruSollu",
                           "909-GeriyeDogruOturma", "910-GeriyeDogruUzanma", "911-GeriyeDogruSagli",
                           "912-GeriyeDogruSollu", "913-YanaSagli", "915-YanaSollu"]
        self.classes = self.ADL_TYPES + self.FALL_TYPES
        self.dataset = {}

        self._load_data()
        self._format_dataset()

    def _load_data(self):
        print("Loading dataset...")
        subdirs = list(filter(os.path.isdir, [os.path.join(self.path, sd) for sd in os.listdir(self.path)]))

        ADLs = []
        FALLs = []
        for i in range(len(subdirs)):
            subdir = subdirs[i]
            print(subdir)
            class_dirs = self._get_subdirs(subdir)
            for class_dir in class_dirs:
                if class_dir.endswith("ADL"):
                    ADLs.append(class_dir)
                elif class_dir.endswith("FALLS"):
                    FALLs.append(class_dir)
                else:
                    print("Error! Wrong motion type.")
        self._process_catogory(ADLs)
        self._process_catogory(FALLs)

    def _process_catogory(self, dirs=[]):
        pass  # TODO
