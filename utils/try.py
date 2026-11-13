import pickle

file_path = "/home/4paradigm/MCANet/Davis_data/p_LM.pkl"

# 使用 pickle 读取数据
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 打印数据类型和内容示例
import pdb;pdb.set_trace()
print(f"Data type: {type(data)}")
print(f"Data sample: {data}")
