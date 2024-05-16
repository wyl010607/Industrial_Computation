import numpy as np
import csv
import os


def load_csv(path,load,label_set,label_dict,domain_dict,data_length,window ):
    dataset = {label:[] for label in label_set}
    for label in label_set:
        csv_path = path+'/'+label_dict[label]+domain_dict[load]+'_2.csv'
        csv_data = csv.reader(open(csv_path))
        csv_data = np.array(list(csv_data))

        start, end = 0, data_length

        # set the endpoint of data sequence
        endpoint = csv_data.shape[0]

        # split the data and transformation
        while end < endpoint:
            sub_data = csv_data[start: end].reshape(-1, )

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

    data, label = np.array(data, dtype="float32"), np.array(label, dtype="int32")

    data = data[:, np.newaxis, :]

    return data, label