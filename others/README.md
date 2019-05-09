# 测试文档

针对 "../recognizer.arcface_recognizer.add_customs" 接口:

* 无embd.npy文件：测试正常文件读入正常，保存embd.npy正常，异常空文件读入正常，数据库正常；
* 有embd.npy文件：测试正常文件读入正常，保存embd.npy正常，异常空文件读入正常，数据库正常；
* 有embd.npy文件，删除其中一人数据：测试正常文件读入正常，保存embd.npy正常，异常空文件读入正常，数据库label补全与扩展正常；


针对 "../display.py" :

* 各按钮槽信号正常，槽函数正常；