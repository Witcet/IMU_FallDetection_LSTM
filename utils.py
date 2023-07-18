import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal

TIME_STEPS = 240


def getsub(dir_path=""):
    return [os.path.join(dir_path, sd) for sd in os.listdir(dir_path)]


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

    def next_batch(self, batch_size=16, training=False, validate=False, evaluate=False, test=False):
        features = self.features
        labels = self.labels

        train_size = int(features.shape[0] * 0.8)
        # valid_size = int(features.shape[0] * 0.9)

        if test:
            pass
        elif training:
            features = features[:train_size]
            labels = labels[:train_size]

        # evaluate是从训练集中剔出来的测试集，但是为了彻底分离训练集与测试集，弃用evaluate
        # elif validate:
        #     features = features[train_size:valid_size]
        #     labels = labels[train_size:valid_size]
        # elif evaluate:
        #     features = features[valid_size:]
        #     labels = labels[valid_size:]

        elif validate:
            features = features[train_size:]
            labels = labels[train_size:]

        else:
            print("Error! Specify one mode!")

        num_samples = features.shape[0]
        n_batches = int(np.floor(num_samples / batch_size))

        for b in tqdm(range(n_batches), file=sys.stdout):
            start = b * batch_size
            end = start + batch_size
            feature_batch = features[start:end]
            label_batch = labels[start:end]

            yield feature_batch, label_batch


class MixData:
    def __init__(self, features, labels):
        print("Mixing dataset above...")
        self.dataset = Data(features, labels, onehot=False)


class SimuDataLoader:
    def __init__(self, path=""):
        self.path = path
        self.UseMot = ["801", "802", "803", "804", "805", "806", "807", "808", "809", "810", "811", "812",
                       "813", "814", "815", "816",
                       "901", "902", "903", "904", "907", "908", "909", "910", "911", "912", "913", "915",
                       "917", "918", "919", "920",
                       '905', '906', '914', '916']
        self.dataset = {}

        self._load_data()
        self._format_dataset()

    def _load_data(self):
        print("Loading dataset...")
        subdirs = getsub(self.path)

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
                # segments = ["torso", "pelvis", "femur_r", "tibia_r"]
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
                    feature_new = feature
                    # interpolate_1d = lambda col: signal.resample(col, TIME_STEPS)
                    # feature_new = np.apply_along_axis(interpolate_1d, 0, feature)
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
                          "809-AyakBurkulmasi", "810-Oksurme", "811-Sandalye", "812-Kanepe",
                          "813-HavayaOturma", "814-YatagaOturma", "815-YatagaUzanma", "816-YataktanKalkma"]
        self.FALL_TYPES = ["901-OneDogruUzanma", "902-OneDogruKorunarakDusme", "903-OneDogruDizlerUzerine",
                           "904-OneDogruDizlerUzerineArdindanUzanma", "907-OneDogruSagli", "908-OneDogruSollu",
                           "909-GeriyeDogruOturma", "910-GeriyeDogruUzanma", "911-GeriyeDogruSagli",
                           "912-GeriyeDogruSollu", "913-YanaSagli", "915-YanaSollu",
                           "905-OneCabukKalkma", "906-OneYavasKalkma", "914-YanaSagliCabukKalkma",
                           "916-YanaSolluCabukKalkma",
                           "917-YataktanDusme", "918-Podyum", "919-Bayilma", "920-BayilmaDuvar"]
        self.classes = self.ADL_TYPES + self.FALL_TYPES
        self.dataset = {}

        self._load_data()
        self._format_dataset()

    def _load_data(self):
        print("Loading dataset...")
        subdirs = getsub(self.path)

        ADLs = []
        FALLs = []
        for i in range(len(subdirs)):
            subdir = subdirs[i]
            print(subdir)
            class_dirs = getsub(subdir)
            for class_dir in class_dirs:
                if class_dir.endswith("ADL"):
                    ADLs.append(class_dir)
                elif class_dir.endswith("FALLS"):
                    FALLs.append(class_dir)
                else:
                    print("Error! Wrong motion type.")
        self._process_category(ADLs)
        self._process_category(FALLs)

    def _process_category(self, dirs=[]):
        for i in tqdm(range(len(dirs)), file=sys.stdout):
            dir_ = dirs[i]
            subdirs = getsub(dir_)

            for subdir in subdirs:
                # print(subdir)
                mot_type = os.path.basename(os.path.normpath(subdir))
                if mot_type not in self.classes:
                    continue

                trials = getsub(subdir)
                for each_trial in trials:
                    data = self._prepare_dataset_from_dir(each_trial)
                    if data.shape[0] == 0:
                        continue
                    try:
                        self.dataset[mot_type] = np.concatenate((self.dataset[mot_type], data), axis=0)
                    except KeyError:
                        self.dataset[mot_type] = data

    def _prepare_dataset_from_dir(self, txt_folder=""):
        imu_files = getsub(txt_folder)

        # ARG 指定实验中IMU传感器的贴放部位，与txt文件一一对应
        imubody_map = {'340540.txt': 'neck',
                       '340539.txt': 'chest',
                       '340537.txt': 'wrist',
                       '340535.txt': 'waist',
                       '340527.txt': 'thigh',
                       '340506.txt': 'shank'}
        body_data = {}
        for imu_file in imu_files:
            imu_index = os.path.basename(imu_file)
            body_data[imubody_map.get(imu_index)] = self._get_imudata(imu_file)

        pertrial_imudata = [[]]

        # ARG 选择哪些部位的IMu数据
        # segments = ["chest", "waist", "thigh", "shank"]
        segments = ["shank"]

        for segment in segments:
            segment_data = body_data[segment]
            try:
                pertrial_imudata = np.concatenate((pertrial_imudata, segment_data), axis=1)
            except ValueError:
                pertrial_imudata = segment_data

        pertrial_imudata = np.expand_dims(pertrial_imudata, axis=0)
        return pertrial_imudata

    def _get_imudata(self, txt_name):
        df = pd.read_csv(txt_name, delimiter=r'\s+', header=4)
        df = df.fillna(method='pad')
        df = df.fillna(method='bfill')

        # ARG 选择IMU的加速度、角速度数据
        # pick_cols = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
        pick_cols = ["Acc_X", "Acc_Y", "Acc_Z"]

        df = df[pick_cols]
        imudata = df.values
        interpolite_1d = lambda col: signal.resample(col, TIME_STEPS)
        new_imudata = np.apply_along_axis(interpolite_1d, 0, imudata)

        # origin_num = imudata.shape[0]
        # new_num = new_imudata.shape[0]
        # during = origin_num / 25
        # t1 = np.arange(0, during, 0.04)
        # t2 = np.arange(0, during, 0.04 * origin_num / new_num)
        # for i in range(6):
        #     plt.plot(t1, imudata[:, i], 'r', t2, new_imudata[:, i], 'g')
        #     plt.show()

        return new_imudata

    def _format_dataset(self):
        print("Formatting dataset...")
        readings = None
        labels = []
        for key in self.dataset:
            if key in self.FALL_TYPES:
                mot_type_num = 1
            else:
                mot_type_num = 0
            features = self.dataset[key]
            num_samples = features.shape[0]
            labels.extend(num_samples * [mot_type_num])
            try:
                readings = np.concatenate((readings, features), axis=0)
            except  ValueError:
                readings = features
        labels = np.array(labels)
        self.dataset = Data(readings, labels)


class ZhjiDataLoader:
    def __init__(self, path=""):
        self.path = path
        self.ADL_TYPES = ["801", "802", "803", "804",
                          "805", "806", "807", "808",
                          "809", "810", "811", "812",
                          "813", "814", "815", "816"]
        self.FALL_TYPES = ["901", "902", "903","900",
                           "904", "907", "908",
                           "909", "910", "911",
                           "912", "913", "915",
                           "905", "906", "914", "916",
                           "917", "918", "919", "920"]
        self.classes = self.ADL_TYPES + self.FALL_TYPES
        self.dataset = {}

        self._load_data()
        self._format_dataset()

    def _load_data(self):
        print("Loading dataset...")
        subdirs = getsub(self.path)

        ADLs = []
        FALLs = []
        for i in range(len(subdirs)):
            subdir = subdirs[i]
            print(subdir)
            class_dirs = getsub(subdir)
            for class_dir in class_dirs:
                if class_dir.endswith("ADL"):
                    ADLs.append(class_dir)
                elif class_dir.endswith("FALLS"):
                    FALLs.append(class_dir)
                else:
                    print("Error! Wrong motion type.")
        self._process_category(ADLs)
        self._process_category(FALLs)

    def _process_category(self, dirs=[]):
        for i in tqdm(range(len(dirs)), file=sys.stdout):
            dir_ = dirs[i]
            subdirs = getsub(dir_)

            for subdir in subdirs:
                print(subdir)
                mot_type = os.path.basename(os.path.normpath(subdir))
                if mot_type not in self.classes:
                    continue

                trials = getsub(subdir)
                for each_trial in trials:
                    data = self._prepare_dataset_from_dir(each_trial)
                    if data.shape[0] == 0:
                        continue
                    try:
                        self.dataset[mot_type] = np.concatenate((self.dataset[mot_type], data), axis=0)
                    except KeyError:
                        self.dataset[mot_type] = data

    def _prepare_dataset_from_dir(self, txt_folder=""):
        imu_files = getsub(txt_folder)

        # ARG 指定实验中IMU传感器的贴放部位，与txt文件一一对应

        # ARG: 佩戴Xsens传感器采集的imu数据
        imubody_map = {'175': 'chest',
                       '1D1': 'calcn',
                       '203': 'wrist',
                       '205': 'shank',
                       '210': 'waist',
                       '2C5': 'thigh',
                       '206': 'head'}
        body_data = {}

        for imu_file in imu_files:
            imu_index = (os.path.basename(imu_file))[-7:-4]
            print(os.path.basename(imu_file))
            body_data[imubody_map.get(imu_index)] = self._get_imudata(imu_file)

        pertrial_imudata = [[]]

        # ARG 选择哪些部位的IMu数据
        # segments = ["chest", "waist", "thigh", "shank"]
        segments = ["shank"]

        for segment in segments:
            segment_data = body_data[segment]
            try:
                pertrial_imudata = np.concatenate((pertrial_imudata, segment_data), axis=1)
            except ValueError:
                pertrial_imudata = segment_data

        pertrial_imudata = np.expand_dims(pertrial_imudata, axis=0)
        return pertrial_imudata

    def _get_imudata(self, txt_name):
        df = pd.read_csv(txt_name, delimiter=r'\s+', header=4)
        df = df.fillna(method='pad')
        df = df.fillna(method='bfill')

        # ARG 选择IMU的加速度、角速度数据
        # pick_cols = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
        pick_cols = ["Acc_X", "Acc_Y", "Acc_Z"]

        df = df[pick_cols]
        df = df[:720:3]
        imudata = df.values
        # interpolite_1d = lambda col: signal.resample(col, TIME_STEPS)
        # new_imudata = np.apply_along_axis(interpolite_1d, 0, imudata)
        return imudata

        # origin_num = imudata.shape[0]
        # new_num = new_imudata.shape[0]
        # during = origin_num / 25
        # t1 = np.arange(0, during, 0.04)
        # t2 = np.arange(0, during, 0.04 * origin_num / new_num)
        # for i in range(6):
        #     plt.plot(t1, imudata[:, i], 'r', t2, new_imudata[:, i], 'g')
        #     plt.show()
        # return new_imudata

    def _format_dataset(self):
        print("Formatting dataset...")
        readings = None
        labels = []
        for key in self.dataset:
            if key in self.FALL_TYPES:
                mot_type_num = 1
            else:
                mot_type_num = 0
            features = self.dataset[key]
            num_samples = features.shape[0]
            labels.extend(num_samples * [mot_type_num])
            try:
                readings = np.concatenate((readings, features), axis=0)
            except  ValueError:
                readings = features
        labels = np.array(labels)
        self.dataset = Data(readings, labels)

# class XsenDataLoader:
#     def __init__(self, path=''):
#         self.path = path
#         self.sensor_num = 4
#         self.dataset = {}
#         self._load_data()
#         self._format_dataset()
#
#     def _load_data(self):
#         print("Loading dataset...")
#         subdirs = getsub(self.path)
#
#         ADLs = []
#         FALLs = []
#         for i in range(len(subdirs)):
#             subdir = subdirs[i]
#             print(subdir)
#             class_dirs = getsub(subdir)
#             for class_dir in class_dirs:
#                 if class_dir.endswith("ADL"):
#                     ADLs = [os.path.join(class_dir, txt_file) for txt_file in os.listdir(class_dir)]
#                 elif class_dir.endswith("FALLS"):
#                     FALLs = [os.path.join(class_dir, txt_file) for txt_file in os.listdir(class_dir)]
#                 else:
#                     print("Error! Wrong motion type.")
#
#         self._process_category(ADLs, "ADL")
#         self._process_category(FALLs, "FALL")
#
#     def _process_category(self, dirs=[], class_=""):
#         if len(dirs) % self.sensor_num != 0:
#             print("Error! The number of TXT files is incorrect.")
#             return
#
#         trial_nums = len(dirs) / self.sensor_num
#
#         for i in range(trial_nums):
#             each_test = dirs[self.sensor_num * i:self.sensor_num * (i + 1)]
#             data = self._prepare_dataset_from_dir(each_test)
#             if data.shape[0] == 0:
#                 continue
#             try:
#                 self.dataset[class_] = np.concatenate((self.dataset[class_], data), axis=0)
#             except KeyError:
#                 self.dataset[class_] = data
#
#     def _prepare_dataset_from_dir(self, each_test):
#         # ARG: 佩戴Xsens传感器采集的imu数据
#         imubody_map = {'175': 'chest',
#                        '1D1': 'waist',
#                        '203': 'thigh',
#                        '205': 'shank',
#                        '210': 'thigh_l',
#                        '2C5': 'shank_l',
#                        '206': 'hand_l'}
#         body_data = {}
#         for imu_file in each_test:
#             imu_index = (os.path.basename(imu_file))[-7:-4]
#             body_data[imubody_map.get(imu_index)] = self._get_imudata(imu_file)
#
#         pertrial_imudata = [[]]
#
#         # ARG: 选择哪些位置的IMU数据
#         for segment in ["chest", "waist", "thigh", "shank"]:
#             segment_data = body_data[segment]
#             try:
#                 pertrial_imudata = np.concatenate((pertrial_imudata, segment_data), axis=1)
#             except ValueError:
#                 pertrial_imudata = segment_data
#
#         pertrial_imudata = np.expand_dims(pertrial_imudata, axis=0)
#         return pertrial_imudata
#
#     def _get_imudata(self, txt_name):
#         df = pd.read_csv(txt_name, delimiter=r'\s+', header=4)
#         df = df.fillna(method="pad")
#         df = df.fillna(method="bfill")
#
#         # ARG 选择IMU的加速度、角速度数据
#         # pick_cols = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
#         pick_cols = ["Acc_X", "Acc_Y", "Acc_Z"]
#
#         df = df[pick_cols]
#         imudata = df.values
#         interpolite_1d = lambda col: signal.resample(col, TIME_STEPS)
#         new_imudata = np.apply_along_axis(interpolite_1d, 0, imudata)
#         return new_imudata
#
#     def _format_dataset(self):
#         print("Formating dataset...")
#         readings = None
#         labels = []
#         for key in self.dataset:
#             if key == "FALL":
#                 mot_type_num = 1
#             else:
#                 mot_type_num = 0
#             features = self.dataset[key]
#             num_samples = features.shape[0]
#             labels.extend(num_samples * [mot_type_num])
#             try:
#                 readings = np.concatenate((readings, features), axis=0)
#             except ValueError:
#                 readings = features
#         labels = np.array(labels)
#         self.dataset = Data(readings, labels)
