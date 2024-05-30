import glob
import numpy as np
import torch


class NetStruc:
    def __init__(
        self,
        dataset,
        list_path,
        train,
        *args,
        **kwargs
    ):
        super().__init__(dataset, list_path, *args, **kwargs)
        self.list_path = list_path
        self.dataset = dataset
        self.train = train
        self.feature_map =None

    def get_feature_map(self, list_path):
        feature_file = open(list_path, 'r')
        feature_list = []
        for ft in feature_file:
            feature_list.append(ft.strip())
        return feature_list

    # 假设图是全连接的
    def get_fc_graph_struc(self, list_path):
        feature_file = open(list_path, 'r')
        struc_map = {}
        feature_list = []
        for ft in feature_file:
            feature_list.append(ft.strip())
        for ft in feature_list:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in feature_list:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)

        return struc_map

    def get_prior_graph_struc(self, list_path, dataset):
        feature_file = open(list_path, 'r')
        struc_map = {}
        feature_list = []
        for ft in feature_file:
            feature_list.append(ft.strip())
        for ft in feature_list:
            if ft not in struc_map:
                struc_map[ft] = []
            for other_ft in feature_list:
                # 根据输入的不同数据集判断相关规则
                if dataset == 'wadi' or dataset == 'wadi2':
                    if other_ft is not ft and other_ft[0] == ft[0]:
                        struc_map[ft].append(other_ft)
                elif dataset == 'swat':
                    if other_ft is not ft and other_ft[-3] == ft[-3]:
                        struc_map[ft].append(other_ft)
        return struc_map

    def get_most_common_features(self, target, all_features, max=3, min=3):
        res = []
        main_keys = target.split('_')
        for feature in all_features:
            if target == feature:
                continue
            f_keys = feature.split('_')
            common_key_num = len(list(set(f_keys) & set(main_keys)))
            if common_key_num >= min and common_key_num <= max:
                res.append(feature)
        return res

    def build_net(self, target, all_features):
        # 获取节点连接和阶段索引关系
        main_keys = target.split('_')
        edge_indexes = [[], []]
        index_feature_map = [target]
        parent_list = [target]
        graph_map = {}
        depth = 2
        for i in range(depth):
            for feature in parent_list:
                children = self.get_most_common_features(feature, all_features)
                if feature not in graph_map:
                    graph_map[feature] = []
                pure_children = []
                for child in children:
                    if child not in graph_map:
                        pure_children.append(child)
                graph_map[feature] = pure_children
                if feature not in index_feature_map:
                    index_feature_map.append(feature)
                p_index = index_feature_map.index(feature)
                for child in pure_children:
                    if child not in index_feature_map:
                        index_feature_map.append(child)
                    c_index = index_feature_map.index(child)
                    edge_indexes[1].append(p_index)
                    edge_indexes[0].append(c_index)
            parent_list = pure_children
        return edge_indexes, index_feature_map

    # 根据连接规则和关联信息构建局部图结构
    def build_loc_net(self, struc, all_features, feature_map=[]):
        index_feature_map = feature_map
        edge_indexes = [[], []]
        for node_name, node_list in struc.items():
            if node_name not in all_features:
                continue
            if node_name not in index_feature_map:
                index_feature_map.append(node_name)
            p_index = index_feature_map.index(node_name)
            for child in node_list:
                if child not in all_features:
                    continue
                if child not in index_feature_map:
                    print(f'error: {child} not in index_feature_map')
                c_index = index_feature_map.index(child)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes

    # 计算节点嵌入
    def construct_data(self, data, feature_map, labels=0):
        res = []
        for feature in feature_map:
            if feature in data.columns:
                res.append(data.loc[:, feature].values.tolist())
            else:
                print(feature, 'not exist in data')
        sample_n = len(res[0])
        if type(labels) == int:
            res.append([labels] * sample_n)
        elif len(labels) == sample_n:
            res.append(labels)
        return res

    def graph_struct(self):
        feature_map = self.get_feature_map(self.list_path)
        fc_struc = self.get_fc_graph_struc(self.dataset)
        fc_edge_index = self.build_loc_net(fc_struc, list(self.train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        return fc_edge_index

    def get_feature(self):
        feature_map = self.get_feature_map(self.list_path)
        return feature_map
