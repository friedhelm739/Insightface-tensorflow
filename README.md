# Insightface-tensorflow

* version 0.3更新中；

>core.config 中相对地址由于调试原因可能出现问题

复现[ArcFace Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)论文，参考了众多复现代码，在此附上链接并对表示感谢~

* https://github.com/deepinsight/insightface
* https://github.com/luckycallor/InsightFace-tensorflow (非常感谢)
* https://github.com/auroua/InsightFace_TF
* https://github.com/tensorflow/models

## 环境依赖

version 0.3:

* ubuntu16.04+2*GTX 1080ti+Python3.6+Anaconda5.2.0+Tensorflow1.7-gpu+MySQL5.7.25

## 结果

|model|lfw|calfw|cplfw|agedb_30|cfp_ff|cfp_fp|lfw_face|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|resnet_v2_m_50|0.979|0.878|0.823|0.886|0.976|0.904|0.866|

>lfw_face 为使用自己的mtcnn模型输出的lfw人脸框制作出来的数据

## 总结

* 在训练准确率曲线逼近100%收敛的情况下，同一模型下自己训练多次的insightface模型eval效果均不如其他人，不知为何；
* 1个epoch后eval就已经收敛了，代码检查无错，存在着疑惑；
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