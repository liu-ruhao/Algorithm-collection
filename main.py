#导入模块
import os
import model as m
import input_data 
import numpy as np
import config as CF
import tensorflow as tf

#生成数据
train, train_label, val, val_label,n_class = input_data.get_files(CF.config["train_dir"], CF.config["ratio"])
CF.config["N_CLASSES"]=n_class
train_batch,train_label_batch=input_data.get_batch(train,train_label,CF.config["IMG_W"],CF.config["IMG_H"],CF.config["BATCH_SIZE"], CF.config["CAPACITY"])

#实例化模型
summary_op=tf.summary.merge_all()
sess=tf.Session()

model=m.model(sess,CF.config,train_batch,train_label_batch)
# 进行batch的训练
try:
    for step in np.arange(CF.config["MAX_STEP"]):
        if model.coord.should_stop():
            break
        tra_loss, tra_acc =  model.train()
        if step % 10 == 0:# 每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            model.summery(summary_op,step)
        if (step+1 )%100 == 0:# 每隔100步，保存一次训练好的模型
            checkpoint_path = os.path.join(CF.config["logs_train_dir"], 'model.ckpt')
            model.saver.save(model.sess, checkpoint_path, global_step=step)
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    model.coord.request_stop()