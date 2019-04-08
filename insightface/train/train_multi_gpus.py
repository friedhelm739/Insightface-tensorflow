import tensorflow as tf
from train.train_tool import arcface_loss,read_single_tfrecord,average_gradients
from core import Arcface_model,config
import time
import os
from evaluate.evaluate import evaluation,load_bin


def train(image,label,train_phase_dropout,train_phase_bn, images_batch, images_f_batch, issame_list_batch):

    train_images_split = tf.split(image, config.gpu_num)
    train_labels_split = tf.split(label, config.gpu_num)      
    
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')    
    scale = int(512.0/batch_size)
    lr_steps = [scale*s for s in config.lr_steps]
    lr_values = [v/scale for v in config.lr_values]
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=lr_values, name='lr_schedule')
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=config.momentum)

    embds = []
    logits = []
    inference_loss = []
    wd_loss = []
    total_train_loss = []
    pred = []
    tower_grads = []
    update_ops = []
    
    for i in range(config.gpu_num):
        sub_train_images = train_images_split[i]
        sub_train_labels = train_labels_split[i]
        
        with tf.device("/gpu:%d"%(i)):
            with tf.variable_scope(tf.get_variable_scope(),reuse=(i>0)):
                
                net, end_points = Arcface_model.get_embd(sub_train_images, train_phase_dropout, train_phase_bn,config.model_params)
                        
                logit = arcface_loss(net,sub_train_labels,config.s,config.m)
                arc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit , labels = sub_train_labels))
                L2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                train_loss = arc_loss + L2_loss
                
                pred.append(tf.to_int32(tf.argmax(tf.nn.softmax(logit),axis=1)))
                tower_grads.append(opt.compute_gradients(train_loss))
                update_ops.append(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
                
                embds.append(net)
                logits.append(logit)
                inference_loss.append(arc_loss)
                wd_loss.append(L2_loss)
                total_train_loss.append(train_loss)

    embds = tf.concat(embds, axis=0)
    logits = tf.concat(logits, axis=0)
    pred = tf.concat(pred, axis=0)
    wd_loss = tf.add_n(wd_loss)/config.gpu_num
    inference_loss = tf.add_n(inference_loss)/config.gpu_num
    
    train_ops = [opt.apply_gradients(average_gradients(tower_grads))]
    train_ops.extend(update_ops)
    train_op = tf.group(*train_ops) 
    
    with tf.name_scope('loss'):
        train_loss = tf.add_n(total_train_loss)/config.gpu_num
        tf.summary.scalar('train_loss',train_loss)    

    with tf.name_scope('accuracy'):
        train_accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, label), tf.float32))
        tf.summary.scalar('train_accuracy',train_accuracy) 
        
    saver=tf.train.Saver(max_to_keep=20)
    merged=tf.summary.merge_all() 
    
    train_images,train_labels=read_single_tfrecord(addr,batch_size,img_size)
    
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run((tf.global_variables_initializer(),
                  tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        writer_train=tf.summary.FileWriter(model_path,sess.graph)
        print("start")
        try:
            for i in range(1,train_step):                
                image_batch,label_batch=sess.run([train_images,train_labels])
                sess.run([train_op,inc_op],feed_dict={image:image_batch,label:label_batch,train_phase_dropout:True,train_phase_bn:True})
                
                if(i%100==0):
                    summary=sess.run(merged,feed_dict={image:image_batch,label:label_batch,train_phase_dropout:True,train_phase_bn:True})
                    writer_train.add_summary(summary,i)
                    
                if(i%1000==0):
                    print('times: ',i)    
#                     print('train_accuracy: ',sess.run(train_accuracy,feed_dict={image:image_batch,label:label_batch,train_phase_dropout:True,train_phase_bn:True}))
#                     print('train_loss: ',sess.run(train_loss,{image:image_batch,label:label_batch,train_phase_dropout:True,train_phase_bn:True}))       
                    print('time: ',time.time()-begin)
                    
                if(i%5000==0):
                    f.write("itrations: %d"%(i)+'\n')
                    for idx in range(len(eval_datasets)):
                        tpr, fpr, accuracy, best_thresholds = evaluation(sess, images_batch[idx], images_f_batch[idx], issame_list_batch[idx], batch_size, img_size, dropout_flag=config.eval_dropout_flag, bn_flag=config.eval_bn_flag, embd=embds, image=image, train_phase_dropout=train_phase_dropout, train_phase_bn=train_phase_bn) 
                        print("%s datasets get %.3f acc"%(eval_datasets[idx].split("/")[-1].split(".")[0],accuracy))
                        f.write("\t %s \t %.3f \t \t "%(eval_datasets[idx].split("/")[-1].split(".")[0],accuracy)+str(best_thresholds)+'\n')
                    f.write('\n')
                    
                if((i>150000)&(i%config.model_save_gap==0)):
                    saver.save(sess,os.path.join(model_path,model_name),global_step=i)
        except  tf.errors.OutOfRangeError:
            print("finished")
        finally:
            coord.request_stop()
            writer_train.close()
        coord.join(threads)
        f.close()
        
            
def main():
    
    with tf.name_scope('input'):
        image = tf.placeholder(tf.float32,[batch_size,img_size,img_size,3],name='image')
        label = tf.placeholder(tf.int32,[batch_size],name='label')
        train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_dropout')
        train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_bn') 

    images_batch = []
    images_f_batch = []
    issame_list_batch = []
    for dataset_path in eval_datasets:
        images, images_f, issame_list = load_bin(dataset_path, img_size)    
        images_batch.append(images)
        images_f_batch.append(images_f)
        issame_list_batch.append(issame_list)
    
    train(image,label, train_phase_dropout, train_phase_bn, images_batch, images_f_batch, issame_list_batch)


if __name__ == "__main__":
    
    img_size = config.img_size
    batch_size = config.batch_size
    addr = config.addrt
    model_name = config.model_name
    train_step = config.train_step
    model_path = config.model_patht
    eval_datasets = config.eval_datasets
    
    begin=time.time()
    
    f = open("./eval_record.txt", 'w')
    f.write("\t dataset \t accuracy \t best_thresholds \t"+'\n')    
    main()
# tensorboard --logdir=/home/dell/Desktop/insightface/model/Arcface_model/