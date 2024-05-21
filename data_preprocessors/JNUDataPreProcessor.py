import numpy as np
from .abs import AbstractDataPreprocessor
from utils.load_csv import load_csv
from utils.load_npz import load_npz
from sklearn.model_selection import train_test_split


class JNUDataPreprocessor(AbstractDataPreprocessor):
    """
    Data Preprocessor for CWRU data.
    """

    def __init__(
            self,
            data_path,
            *args,
            **kwargs
    ):
        """
        Initialize the CWRU Data Preprocessor.

        Parameters
        ----------
        data_path : str
            Path to the data file.
        s_load : str
            source domain working condition.
        t_load : str
            target domain working condition.
        s_label_set : list
            source domain label set.
        t_label_set : list
            target domain label set.
        feature_name : str
            name of feature column.
        name_dict : dict
            name dict of data.
        train_ratio : float, optional
            Ratio of training data. Default is 0.6.
        valid_ratio : float, optional
            Ratio of validation data. Default is 0.2.
        """

        super().__init__(data_path, *args, **kwargs)
        self.s_load = kwargs["s_load"]
        self.t_load = kwargs["t_load"]
        self.s_label_set = kwargs["s_label_set"]
        self.t_label_set = kwargs["t_label_set"]
        self.label_dict = kwargs["label_dict"]
        self.domain_dict = kwargs["domain_dict"]
        self.data_length = kwargs["data_length"]
        self.window = kwargs["window"]
        self.s_load = kwargs["s_load"]
        self.t_load = kwargs["t_load"]
        self.random_seed = kwargs["random_seed"]
        self.s_data = None
        self.s_label = None
        self.t_data = None
        self.t_label = None
        self.data_dimension = kwargs["data_dimension"]
        self.class_num = kwargs["class_num"]
        self.update_model_params = {
            'dimension': self.data_dimension,
            'class_num': self.class_num
        }
        self.load_data()

    def load_data(self):
        """
        Load data from the specified file path.

        This method loads the data, time stamps, and variable index dictionary
        from a file at `self.data_path`.
        """

        #s_path = self.data_path + "t/Drive_end_" + str(self.s_load) + "/"
        ##t_path = self.data_path + "/Drive_end_" + str(self.t_load) + "/"
        if self.data_dimension == 'DWT':
            s_path = self.data_path+"/DWT/"+str(self.s_load)
            t_path = self.data_path+"/DWT/"+str(self.t_load)
            self.s_data,self.s_label = load_npz(s_path,self.s_load,self.s_label_set,self.name_dict)
            self.t_data,self.t_label = load_npz(t_path,self.t_load,self.t_label_set,self.name_dict)
        # 加载源域数据
        else:
            self.s_data, self.s_label = load_csv(self.data_path, self.s_load, self.s_label_set,self.label_dict,self.domain_dict, self.data_length, self.window)                                        
            # 加载目标域数据
            self.t_data, self.t_label = load_csv(self.data_path, self.t_load, self.t_label_set,self.label_dict,self.domain_dict, self.data_length, self.window)
        
    def preprocess(self):
        """
        Preprocess the loaded data.

        This method extracts specific indices from the data based on the process,
        control, and disturb variable lists, and prepares it for further processing.
        by the way, load the adj_mx and add it to the update_model_params

        Returns
        -------
        np.ndarray
            The preprocessed data array.
        """
        s_preprocessed_data = self.s_data
        t_preprocessed_data = self.t_data

        return s_preprocessed_data, self.s_label, t_preprocessed_data, self.t_label

    def split_data(self, preprocessed_data):
        """
        Split the preprocessed data into training, validation, and testing sets.

        Parameters
        ----------
        preprocessed_data : tuple
            The preprocessed data array to be split.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing the training, validation, and test data arrays, respectively.
        """
        s_train_data = preprocessed_data[0]
        s_train_label = preprocessed_data[1]
        t_train_data, t_val_data, t_train_label, t_val_label = train_test_split(preprocessed_data[2],
                                                                                preprocessed_data[3],
                                                                                test_size=1 - self.train_ratio,
                                                                                random_state=self.random_seed)
        t_val_data, t_test_data, t_val_label, t_test_label = train_test_split(t_val_data, t_val_label,
                                                                              test_size=1 - self.valid_ratio)

        return s_train_data, s_train_label, t_train_data, t_train_label, t_val_data, t_val_label, t_test_data, t_test_label
