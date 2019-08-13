import os
import glob
import config as CF
import numpy as np
import tensorflow as tf

#通过将训练集按照指定格式放入input_data文件夹下，运行get_file函数，给出训练集和测试集图片的路径。
def get_files(file_dir="./input_data/",ratio=CF.config["ratio"]):
    print("调用数据get_file")
    if glob.glob(file_dir+"label2id")!=[]:
        print("删除遗留的label2id文件")
        os.remove(file_dir+"label2id")
    label_files=glob.glob(file_dir+"*")
    #print(label_files)
    label_lst=[x.split("\\")[-1] for x in label_files]
    total_data=[]
    t_label2id={}
    #print(label_lst)
    for idd,label in enumerate(label_lst):
        temp={}
        temp["id"]=int(idd)
        temp["num"]=0
        t_label2id[label]=temp
        path_label=file_dir+label
        path_image=glob.glob(path_label+"/*")
        #print(path_image)
        for path in path_image:
            total_data.append([path,label])
            #print(total_data)
            t_label2id[label]["num"]+=1
    print("数据一共有%s个，一共有%s个类别，其中:"%(len(total_data),len(label_lst)))
    with open(file_dir+"label2id","w",encoding="utf-8") as f:
        for label,temp in t_label2id.items():
            idd=temp["id"]
            num=temp["num"]
            print("类别%s的id是%s，有样本%s个！"%(label,idd,num))
            f.write("%s\t%s\t%s\n"%(label,idd,num))
    train_data=[]
    test_data=[]
    for path_label in total_data:#?
        path,label=path_label
        if np.random.random()>ratio:
            test_data.append([path,t_label2id[label]["id"]])
        else:
            train_data.append([path,t_label2id[label]["id"]])
    tra_images = [x[0] for x in train_data]
    tra_labels = [x[1] for x in train_data]
    val_images = [x[0] for x in test_data]
    val_labels = [x[1] for x in test_data]
    n_class=len(label_lst)
    #print(label_lst)
    #print(len(label_lst))
    #print(tra_images,tra_labels,val_images,val_labels)
    return tra_images,tra_labels,val_images,val_labels,n_class
    

#=通过路径将图片转化成数字输入
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int32)
    input_queue=tf.train.slice_input_producer([image,label])#?
    #input_queue = tf.data.Dataset.from_tensor_slices([image, label])
    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])
    image=tf.image.decode_jpeg(image_contents,channels=3)
    image=tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image=tf.image.per_image_standardization(image)
    
    image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,num_threads=32,capacity=capacity)
    
    #重新排列label，行数为[batch_size]
    label_batch=tf.reshape(label_batch,[batch_size])
    image_batch=tf.cast(image_batch,tf.float32)
    #print(image_batch)
    return image_batch,label_batch
   

if __name__=="__main__":
    train,train_label,val,val_label,n_class=get_files(file_dir="./input_data/",ratio=CF.config["ratio"])
    train_batch, train_label_batch = get_batch(train, train_label, 64, 64, 20, 200)
    print(train_batch)
    print(train_label_batch)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        