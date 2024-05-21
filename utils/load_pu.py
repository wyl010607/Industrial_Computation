import os
import numpy as np
from scipy.io import loadmat
def load_pu(data_path, label_set, work_state, name_dict, data_length, window):
    dataset = {label: [] for label in label_set}
    for label in label_set:
        for rar_name in name_dict[label]:
            files = os.listdir(data_path + "/" + rar_name)
            for file_name in files:
                if file_name.startswith(work_state):
                    mat_data = loadmat(data_path + "/" + rar_name + "/" + file_name)

                    # 获取MAT文件中的变量名
                    variable_names = mat_data.keys()

                    # MAT文件中的变量名：dict_keys(['__header__', '__version__', '__globals__', 'N15_M07_F04_K001_4'])
                    # eg. N15_M07_F04_K001_4
                    data_name = list(variable_names)[3]

                    data_temp = mat_data[data_name][0][0][2][0][6][2]
                    start, end = 0, data_length
                    endpoint = data_temp.shape[1]

                    while end < endpoint:
                        sub_data = data_temp[0][start: end].reshape(-1, )
                        dataset[label].append(sub_data)
                        start += window
                        end += window
        dataset[label] = np.array(dataset[label], dtype="float32")

    data, label = [], []
    for key in dataset.keys():
        label.append(np.tile(key, dataset[key].shape[0]))
        data.append(dataset[key])

    data = np.concatenate(data)
    label = np.concatenate(label)
    print(label)
    data, label = np.array(data, dtype="float32"), np.array(label, dtype="int32")

    data = data[:, np.newaxis, :]

    return data, label