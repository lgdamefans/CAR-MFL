import pickle
import os
import pandas as pd

class PartitionTester:
    def __init__(self, partition_dir='partitions'):
        self.partition_dir = partition_dir

    def list_pkl_files(self):
        """列出所有pkl文件"""
        pkl_files = [f for f in os.listdir(self.partition_dir) if f.endswith('.pkl')]
        print(f"发现的pkl文件: {pkl_files}")
        return pkl_files

    def analyze_partition(self, file_name):
        """分析单个分区文件的内容"""
        file_path = os.path.join(self.partition_dir, file_name)
        print(f"\n分析文件: {file_name}")
        print("-" * 50)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 基本信息
        print(f"数据类型: {type(data)}")
        print(f"键列表: {list(data.keys())}")

        # 详细分析每个分区
        for key, value in data.items():
            print(f"\n{key}分区:")
            print(f"类型: {type(value)}")
            print(f"数据量: {len(value)}")

            # 根据数据类型不同处理方式不同
            if isinstance(value, list):
                print(f"前5个索引: {value[:5] if len(value) >= 5 else value}")
            elif isinstance(value, dict):
                print(f"字典键: {list(value.keys())[:5] if len(value.keys()) >= 5 else list(value.keys())}")
                # 分析第一个客户端的数据
                if len(value) > 0:
                    first_client = list(value.keys())[0]
                    print(f"客户端 {first_client} 的数据:")
                    client_data = value[first_client]
                    print(f"  数据类型: {type(client_data)}")
                    if isinstance(client_data, dict):
                        print(f"  键列表: {list(client_data.keys())}")
                        for client_key, client_value in client_data.items():
                            print(f"  {client_key}: 类型 {type(client_value)}, 长度 {len(client_value)}")
                            if isinstance(client_value, list) and len(client_value) > 0:
                                print(f"    前5个元素: {client_value[:5] if len(client_value) >= 5 else client_value}")

        return data

def main():
    tester = PartitionTester()
    pkl_files = tester.list_pkl_files()

    # 分析每个文件
    for pkl_file in pkl_files:
        data = tester.analyze_partition(pkl_file)

if __name__ == "__main__":
    main()