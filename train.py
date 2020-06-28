import os
import time
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from Unet import Unet
from data_loader import DataLoader
import mir_eval

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/work/u4851006/music/keras-unet-vocal-separation/data/processed_data', 'Link to dataset directory.')
flags.DEFINE_string('data_dir2', '/work/u4851006/music/keras-unet-vocal-separation/data/MIR-1K_resized', 'Link to second dataset directory.')
flags.DEFINE_string('save_dir', './checkpoints/checkpoints_corpus_mir1k_2', 'Link to checkpoints directory.')
flags.DEFINE_string('train_list', './lists/train_list.txt', 'Link to training list.')
flags.DEFINE_string('train_list2', './lists/train_list2.txt', 'Link to training list.')
flags.DEFINE_string('val_list', './lists/val_list.txt', 'Link to val list.')

flags.DEFINE_integer('batch_size', 8, None)
flags.DEFINE_float('lr', 0.0001, None)
flags.DEFINE_integer('num_steps', 1200000, None)

flags.DEFINE_integer('steps_per_save', 2000, None)
flags.DEFINE_integer('steps_per_eval', 2000, None)
flags.DEFINE_integer('log_freq', 50, None)

@tf.function
def train_step(model, inputs, gt):
    logging.info('Tracing, in Func. "train_step" ...')
    mask = model(inputs)

    #scale just in case
    _, gt_height, gt_width, _ = tf.unstack(tf.shape(gt))
    _, pred_height, pred_width, _ = tf.unstack(tf.shape(mask))
    scaled_gt = tf.image.resize(gt, tf.shape(mask)[1:3], method=tf.image.ResizeMethod.BILINEAR)
    scaled_gt /= tf.cast(gt_height / pred_height, dtype=tf.float32)
    #calculate mean of absolute difference (l1 norm)
    preds = tf.multiply(inputs,mask) #mask
    l1_norm = tf.reduce_mean(tf.reduce_sum(tf.norm(preds - gt, ord=1, axis=3), axis=(1, 2)))
    loss = l1_norm
    
    return preds, loss

def eval_step(model, inputs, gt):
    mask = model(inputs)
    preds = tf.multiply(inputs,mask)
    preds = np.expand_dims(tf.squeeze(preds).numpy().flatten(),axis=0)
    gt = np.expand_dims(tf.squeeze(gt).numpy().flatten(),axis=0)
    inputs = np.expand_dims(tf.squeeze(inputs).numpy().flatten(),axis=0)
    #Se, Sr
    (SDR, SIR, SAR, _) = mir_eval.separation.bss_eval_sources(gt,preds)
    #Sm, Sr
    (SDR2, _, _, _) = mir_eval.separation.bss_eval_sources(gt,inputs)
    NSDR = SDR - SDR2 #SDR(Se, Sr) − SDR(Sm, Sr)

    return SDR,SIR,SAR,NSDR

def write_summary(summary_writer, step, metric, mode, input=None, preds=None, gt=None):
    with summary_writer.as_default():  
        if mode == 'training':
            tf.summary.scalar('training_loss', metric.result(), step=step)

def val_write_summary(summary_writer, step, SDR_metric, SIR_metric, SAR_metric, NSDR_metric, mode='validation'):
    with summary_writer.as_default():  
        if mode == 'validation':
            tf.summary.scalar('SDR_metric', SDR_metric.result(), step=step)
            tf.summary.scalar('SIR_metric', SIR_metric.result(), step=step)
            tf.summary.scalar('SAR_metric', SAR_metric.result(), step=step)
            tf.summary.scalar('NSDR_metric', NSDR_metric.result(), step=step)

def main(argv):
    ''' Prepare dataset '''
    data_loader = DataLoader(FLAGS.data_dir, FLAGS.data_dir2, FLAGS.train_list,FLAGS.train_list2, FLAGS.val_list)
    train_dataset, val_dataset = data_loader.create_tf_dataset(flags=FLAGS)

    ''' Declare and setup optimizer '''
    num_steps = FLAGS.num_steps
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
    logging.info('Training steps: {}'.format(num_steps))

    ''' Create metric and summary writers '''
    train_metric = tf.keras.metrics.Mean(name='train_loss')
    SDR_metric =  tf.keras.metrics.Mean(name='SDR')
    SIR_metric =  tf.keras.metrics.Mean(name='SIR')
    SAR_metric =  tf.keras.metrics.Mean(name='SAR')
    NSDR_metric = tf.keras.metrics.Mean(name='NSDR')
    time_metric = tf.keras.metrics.Mean(name='elapsed_time_per_step')
    train_summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.save_dir, 'summaries', 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.save_dir, 'summaries', 'val'))

    ''' Initialize model '''
    unet = Unet()
    
    ''' Check if there exists the checkpoints '''
    ckpt_path = os.path.join(FLAGS.save_dir, 'tf_ckpt')
    ckpt = tf.train.Checkpoint(optimizer=optimizer, net=unet)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=20)

    if manager.latest_checkpoint:
        logging.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")

    status = ckpt.restore(manager.latest_checkpoint).expect_partial()
    
    ''' Start training '''
    step = optimizer.iterations.numpy()
    while step < num_steps:
        for data in train_dataset:
            mix, vocal = tf.split(data, [1, 1], -1) #16, 512, 128
            # Model inference and use 'tf.GradientTape()' to trace gradients.
            start_time = time.time()
            with tf.GradientTape() as tape:
                preds, loss = train_step(model=unet,inputs=mix, gt=vocal)
            # Update weights. Compute gradients and apply to the optimizersr.
            grads = tape.gradient(loss, unet.trainable_variables)
            optimizer.apply_gradients(zip(grads, unet.trainable_variables))
            elapsed_time = time.time() - start_time
            
            # Logging
            train_metric.update_state(loss)
            time_metric.update_state(elapsed_time)
            step = optimizer.iterations.numpy()
            if step % FLAGS.log_freq == 0:
                #if step % 1 == 0:
                write_summary(summary_writer=train_summary_writer, step=step,
                                metric=train_metric, mode='training',
                                input=mix, preds=preds, gt=vocal)
                
                logging.info('Step {:>7}, Training Loss: {:.5f}, ({:.3f} sec/step)'.format(step,
                                                                                    train_metric.result(), 
                                                                                    time_metric.result()))
                train_metric.reset_states()
                time_metric.reset_states()
            # Evaluate
            if step % FLAGS.steps_per_eval == 0:
                #if step % 1 == 0:
                for data in val_dataset:
                    mix, vocal = tf.split(data, [1, 1], -1)
                    SDR,SIR,SAR,NSDR = eval_step(model=unet, inputs=mix, gt=vocal)
                    SDR_metric.update_state(SDR)
                    SIR_metric.update_state(SIR)
                    SAR_metric.update_state(SAR)
                    NSDR_metric.update_state(NSDR)

                val_write_summary(summary_writer=val_summary_writer, step=step,
                    SDR_metric=SDR_metric, SIR_metric=SIR_metric, SAR_metric=SAR_metric, NSDR_metric=NSDR_metric,
                    mode='validation')

                logging.info('*****Steps {:>7}, SDR = {:.5f}, SIR = {:.5f}, SAR = {:.5f}, NSDR = {:.5f}*****'.format(step, SDR_metric.result(),SIR_metric.result(),SAR_metric.result(),NSDR_metric.result()))
                SDR_metric.reset_states()
                SIR_metric.reset_states()
                SAR_metric.reset_states()
                NSDR_metric.reset_states()

            # Save checkpoints
            if step % FLAGS.steps_per_save == 0:
                #if step % 1 == 0:
                manager.save(checkpoint_number=step)
                logging.info('*****Steps {:>7}, save checkpoints!*****'.format(step))

if __name__ == '__main__':
    app.run(main)