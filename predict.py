from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import argparse
import sys,os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import errno 

def load_image(filename):
    #Read in the image_data to be classified."""
    return tf.gfile.FastGFile(filename, 'rb').read()

def load_labels(filename):
    #Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]

def load_graph(filename):
    #Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def run_graph(src, dest, labels, input_layer_name, output_layer_name, num_top_predictions):
    with tf.Session() as sess:
        i=0
        for f in os.listdir(src):
            new_f=os.path.splitext(f)[0]
            im=Image.open(os.path.join(src,f))
            #print("im is ",im)
            img=im.convert('RGB')
            img.save(os.path.join(dest,new_f+'.jpg'))
            image_data=load_image(os.path.join(dest,new_f+'.jpg'))
            softmax_tensor=sess.graph.get_tensor_by_name(output_layer_name)
            predictions,=sess.run(softmax_tensor, {input_layer_name: image_data})
            
            # Sort to show labels in order of confidence             
            top_k = predictions.argsort()[-num_top_predictions:][::-1]
            for node_id in top_k:
                predicted_label = labels[node_id]
                score = predictions[node_id]
                print(new_f+',',predicted_label)
                #outfile.write(test[i]+','+human_string+'\n')
            i+=1
        print("done")

def main():
    src=sys.argv[1]
    print(src)
    dest=os.path.join('./data_bully','test_data_00')
    mkdir_p(dest)
    labels='./tmp/output_labels.txt'
    graph='./tmp/output_graph.pb'
    input_layer='DecodeJpeg/contents:0'
    output_layer='final_result:0'
    num_top_predictions=1
    labels = load_labels(labels)
    load_graph(graph)
    run_graph(src,dest,labels,input_layer,output_layer,num_top_predictions)

if __name__ == '__main__':
    main()
