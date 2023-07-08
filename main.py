import os
import pickle
import numpy as np
import warnings
from utils import RealDataLoader,SimuDataLoader,XsenDataLoader,MixData

warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

TB_DIR="for_tensorboard"
model_dir="saved_model"

class FallDetector:
    def __init__(self,batch_size,num_features,timesteps,num_classes):
        self.batch_size=batch_size
        self.num_features=num_features
        self.timesteps=timesteps
        self.num_classes=num_classes

        os.makedirs(model_dir,exist_ok=True)
        self.model_file=os.path.join(model_dir,"model.ckpt")

        print("Resetting the default graph...")
        self.build_model()
        self.define_loss()
        self.define_optimizer()

        self.saver=tf.train.Saver()
        self.tensorboard_op=tf.summary.merge_all()
        print("Model built!")

    def build_model(self):
        # Before beginning, reset the graph.
        tf.compat.v1.reset_default_graph()

        # define placeholders
        self.x=tf.compat.v1.placeholder(tf.float32,shape=(self.batch_size,self.timesteps,self.num_features),name='x')
        self.y=tf.compat.v1.placeholder(tf.float32,shape=(self.batch_size,self.num_classes),name='y')
        self.is_training=tf.compat.v1.placeholder(tf.bool,name='is_training')

        # normalize input
        normalized_input=tf.layers.batch_normalization(self.x,axis=2,training=self.is_training)

        # convert input tensor to list of tensors
        inputs=tf.split(normalized_input,self.timesteps,axis=1)
        inputs=[tf.squeeze(inp) for inp in inputs]

        # RNN layers
        rnn1_out=self._lstm_layer(inputs,32,"lstm-1")
        rnn2_out=self._lstm_layer(rnn1_out,64,"lstm-2")

        # Get last node
        final_node=rnn2_out[-1]

        # Dense layers
        dense1_out=self._dense_layer(final_node,32,"dense-1",activation=tf.nn.elu)
        dense2_out=self._dense_layer(dense1_out,self.num_classes,"dense-2",activation=tf.nn.softmax)

        self.prefiction=dense2_out

        # Accuracy
        pred_classes=tf.argmax(self.prefiction,axis=-1)
        actual_classes=tf.argmax(self.y,axis=-1)
        correct = tf.equal(pred_classes,actual_classes)
        self.accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name='accuracy')

    def define_loss(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,logits=self.prefiction))
        tf.summary.scalar("Cross-entropy-loss",self.loss)

    def define_optimizer(self):
        optimizer=tf.train.AdamOptimizer(0.0001)
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step=optimizer.minimize(self.loss)

    def _lstm_layer(self,inputs,num_units,scope,cell_type="LSTM"):
        with tf.variable_scope(scope):
            if cell_type=="LSTM":
                lstm_cell=tf.nn.rnn_cell.LSTMCell(num_units)
            elif cell_type=="GRU":
                lstm_cell = tf.nn.rnn_cell.GRUCell(num_units)
            else:
                raise NotImplementedError("{} type not implemented.".format(cell_type))

            outputs,states=tf.nn.static_rnn(lstm_cell,inputs,dtype=tf.float32)
        return outputs

    def _dense_layer(self,inputs,num_units,scope,activation=None):
        with tf.variable_scope(scope):
            w=tf.get_variable("w",shape=(inputs.shape[-1].value,num_units))
            b=tf.get_variable("b",shape=(num_units),initializer=tf.constant_initializer(0.1))

            out=tf.matmul(inputs,w)+b
            if activation is not None:
                out=activation(out)

        return out

    def train(self,path,epochs=50,log_step=10,resume=False,dataloader=""):
        print("Beginning the training process...")

        loader_class =  {"simu":SimuDataLoader(path),
                        "real":RealDataLoader(path),
                        "xsen":XsenDataLoader(path)}

        if not path:
            if not os.path.isfile("train_dataset.pkl"):
                print("Error!No dataset to load.")

        if os.path.isfile("train_dataset.pkl"):
            data=pickle.load(open("train_dataset.pkl",'rb'))
        elif dataloader in loader_class.keys():
            data=loader_class.get(dataloader)
            pickle.dump(data,open("train_dataset.pkl",'wb'))
        else:
            print("Error!Unknown data loader.")

















