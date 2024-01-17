import numpy as np
from .abs import AbstractDataPreprocessor


class ASPENDataPreprocessor(AbstractDataPreprocessor):
    """
    Data Preprocessor for ASPEN data.
    """

    def __init__(
        self, data_path, input_vars_list=None, output_vars_list=None, *args, **kwargs
    ):
        """
        Initialize the ASPEN Data Preprocessor.

        Parameters
        ----------
        data_path : str
            Path to the data file.
        train_ratio : float, optional
            Ratio of training data. Default is 0.6.
        valid_ratio : float, optional
            Ratio of validation data. Default is 0.2.
        input_vars_list : list of str, optional
            List of process variable names. Default is an empty list.
        output_vars_list : list of str, optional
            List of control variable names. Default is an empty list.
        """

        super().__init__(data_path, *args, **kwargs)
        self.data = None
        self.header = None
        self.input_vars_list = input_vars_list if input_vars_list is not None else []
        self.output_vars_list = output_vars_list if output_vars_list is not None else []

        self.load_data()

    def load_data(self):
        """
        Load data from the specified file path.
        """
        with open(self.data_path, "r") as f:
            self.header = f.readline().strip().split(",")
        self.data = np.loadtxt(self.data_path, delimiter=",", skiprows=1)

    def preprocess(self):
        """
        Preprocess the data.
        """
        indices = [
            self.header.index(var)
            for var in (self.input_vars_list + self.output_vars_list)
        ]
        preprocessed_data = self.data[:, indices]
        preprocessed_data = self._drop_rows_have_missing_values(preprocessed_data)
        input_index_list = [self.header.index(var) for var in self.input_vars_list]
        output_index_list = [self.header.index(var) for var in self.output_vars_list]
        self.update_dataset_params = {
            "input_index_list": input_index_list,
            "output_index_list": output_index_list,
        }
        self.update_trainer_params = {
            "input_index_list": input_index_list,
            "output_index_list": output_index_list,
        }
        self.update_model_params = {
            "input_dim": len(input_index_list),
            "output_dim": len(output_index_list),
        }

        return preprocessed_data

    def split_data(self, preprocessed_data):
        """
        Split the preprocessed data into different sets.

        Parameters
        ----------
        preprocessed_data : np.ndarray
            Preprocessed data.

        Returns
        -------
        tuple of np.ndarray
            Training, validation and testing data.
        """
        total_length = len(preprocessed_data)
        train_end = int(self.train_ratio * total_length)
        valid_end = int((self.train_ratio + self.valid_ratio) * total_length)

        train_data = preprocessed_data[:train_end]
        valid_data = preprocessed_data[train_end:valid_end]
        test_data = preprocessed_data[valid_end:]

        return train_data, valid_data, test_data

    def _drop_rows_have_missing_values(self, data):
        """
        Check if there are missing values(None, Nan, Null, inf ) in the data.
        if any ,delete the row
        """
        mask = np.isnan(data).any(axis=1) | np.isinf(data).any(axis=1)
        data = data[~mask]
        return data
