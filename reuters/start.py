import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行就是使用cpu，不加上就是使用gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
