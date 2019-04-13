# Insightface-tensorflow

* version 0.3更新中；

复现[ArcFace Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)论文，参考了众多复现代码，在此附上链接并对表示感谢~

* https://github.com/deepinsight/insightface
* https://github.com/luckycallor/InsightFace-tensorflow (非常感谢)
* https://github.com/auroua/InsightFace_TF
* https://github.com/tensorflow/models

## 环境依赖

version 0.3:

* ubuntu16.04 + 2*GTX 1080ti + Python3.6 + Anaconda5.2.0 + Tensorflow1.7-gpu + MySQL5.7.25

## 结果

|model|lfw|calfw|cplfw|agedb_30|cfp_ff|cfp_fp|lfw_face|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|resnet_v2_m_50(prelu)|0.979|0.878|0.823|0.886|0.976|0.904|0.866|
|resnet_v2_m_50(leaky_relu)|0.992|0.932|0.860|0.935|0.990|0.910|0.943|

>lfw_face 为使用自己的mtcnn模型输出的lfw人脸框制作出来的数据

## 总结

* 模型其他结构未动，更换了激活函数prelu替换leaky_relu就能降低最高5%的准确率，relu未实验，而源码用的就是prelu，这就奇了怪了；
* 识别准确率受到人脸检测的极大干扰；

## 人脸识别系统

人脸识别器在recognizer文件夹下，人脸检测基于我的[MTCNN](https://github.com/friedhelm739/MTCNN-tensorflow)仓库；

人脸识别器接口概览：

>基础接口：

* get_embd ： 获取人脸特征接口，shape=[n,512]；
* align_face ： 对齐人脸接口；
* recognize ： 人脸识别接口，输出识别结果与人脸框；

>扩展接口(update in future version)：

* add_customs ： 向数据库内添加新增人员或增加现有人员数据接口；
* add_embds ： 向数据库内增加现有人员数据接口；
* update_customs ： 向数据库内更新现有人员数据接口(旧数据会被替换)；
* del_customs : 删除数据库中指定成员的全部信息；