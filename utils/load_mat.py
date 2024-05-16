import numpy as np
from scipy.io import loadmat


def load_mat(path, load, label_set, data_dict, feature_name, data_length, window):
    dataset = {label: [] for label in label_set}
    for label in label_set:
        fault_type = data_dict[load][label]
        
        if fault_type < 100:
            axis = "X0" + str(fault_type) + feature_name
        else:
            axis = "X" + str(fault_type) + feature_name
        if fault_type == 174:
            axis = "X" + str(fault_type-1) + feature_name
        
        mat_data = loadmat(path + str(fault_type) + ".mat")[axis]

        start, end = 0, data_length

        # set the endpoint of data sequence
        endpoint = mat_data.shape[0]

        # split the data and transformation
        while end < endpoint:
            sub_data = mat_data[start: end].reshape(-1, )

            dataset[label].append(sub_data)
            start += window
            end += window
    for label in label_set:
        dataset[label] = np.array(dataset[label], dtype="float32")

    data, label = [], []

    for key in dataset.keys():
        label.append(np.tile(key, dataset[key].shape[0]))
        data.append(dataset[key])

    data = np.concatenate(data)
    label = np.concatenate(label)

    data, label = np.array(data, dtype="float32"), np.array(label, dtype="int32")

    data = data[:, np.newaxis, :]

    return data, label
