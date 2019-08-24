from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from autoaugment import wrn
import data

batch_size = 32
ratio = 5
steps = 400000
warmup_steps = 20000
lr = 0.03
min_lr_ratio = 0.004


def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
    step_ratio = tf.to_float(global_step) / tf.to_float(num_train_steps)
    if schedule == "linear_schedule":
        coeff = step_ratio
    elif schedule == "exp_schedule":
        scale = 5
        # [exp(-5), exp(0)] = [1e-2, 1]
        coeff = tf.exp((step_ratio - 1) * scale)
    elif schedule == "log_schedule":
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        coeff = 1 - tf.exp((-step_ratio) * scale)
    return coeff * (end - start) + start


def anneal_sup_loss(sup_logits, sup_labels, sup_loss, global_step):
    tsa_start = 1. / 10
    eff_train_prob_threshold = get_tsa_threshold(
        'linear_schedule', global_step, steps,
        tsa_start, end=1)

    one_hot_labels = tf.one_hot(
        sup_labels, depth=10, dtype=tf.float32)
    sup_probs = tf.nn.softmax(sup_logits, axis=-1)
    correct_label_probs = tf.reduce_sum(
        one_hot_labels * sup_probs, axis=-1)
    larger_than_threshold = tf.greater(
        correct_label_probs, eff_train_prob_threshold)
    loss_mask = 1 - tf.cast(larger_than_threshold, tf.float32)
    loss_mask = tf.stop_gradient(loss_mask)
    sup_loss = sup_loss * loss_mask
    avg_sup_loss = tf.divide(
        tf.reduce_sum(sup_loss),
        tf.maximum(tf.reduce_sum(loss_mask), 1),
        name='sup_loss_tensor')
    tsa_threshold = tf.divide(
        eff_train_prob_threshold,
        1.0,
        name='tsa_threshold_tensor'
    )
    return sup_loss, avg_sup_loss, tsa_threshold


def decay_weights(cost, weight_decay_rate):
    """Calculates the loss for l2 weight decay and adds it to `cost`."""
    costs = []
    for var in tf.trainable_variables():
        costs.append(tf.nn.l2_loss(var))
    cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
    return cost


def kl_divergence(p_logits, q_logits):
    p = tf.nn.softmax(p_logits)
    log_p = tf.nn.log_softmax(p_logits)
    log_q = tf.nn.log_softmax(q_logits)

    return tf.reduce_sum(p * (log_p - log_q), -1)


def model_fn(features, labels, mode, params, config):
    # print("============calling model_fn================")
    sup_only = params['sup_only']
    # print(features)
    if mode == tf.estimator.ModeKeys.EVAL:
        all_data = features
    else:
        sup_x = features['image']
        sup_y = features['label']
        sup_batch_size = sup_x.shape[0]
        unsup = labels['unsup']
        aug = labels['aug']
        unsup_batch_size = unsup.shape[0]
        all_data = tf.concat([
            sup_x,
            unsup,
            aug
        ], axis=0)

    logits = wrn.build_wrn_model(all_data, params['n_classes'], 32)
    # print(np.shape(logits))
    predicted_classes = tf.argmax(logits, axis=-1, output_type=tf.int32)
    probs = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        sup_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        sup_loss = tf.reduce_mean(sup_loss)
        accuracy = tf.metrics.accuracy(
            labels, predicted_classes, name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode, loss=sup_loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,
            'probs': probs,
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    assert mode == tf.estimator.ModeKeys.TRAIN
    # print(sup_loss.shape)
    sup_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=sup_y, logits=logits[:sup_batch_size])
    sup_loss = tf.reduce_mean(sup_loss, name='sup_loss_tensor')
    # sup_loss, avg_sup_loss, tsa_threshold = anneal_sup_loss(
    #     logits[:sup_batch_size],
    #     labels[:sup_batch_size],
    #     sup_loss,
    #     tf.train.get_global_step()
    # )
    # sup_loss = avg_sup_loss
    if sup_only:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            sup_loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode, loss=sup_loss, train_op=train_op)
    unsup_loss = kl_divergence(
        tf.stop_gradient(
            logits[sup_batch_size: sup_batch_size + unsup_batch_size]),
        logits[sup_batch_size + unsup_batch_size:]
    )
    unsup_loss = tf.reduce_mean(unsup_loss, name='unsup_loss_tensor')
    total_loss = sup_loss + unsup_loss
    total_loss = decay_weights(total_loss, 5e-4)

    metric_dict = {
        'sup_loss': 'sup_loss_tensor',
        'unsup_loss': 'unsup_loss_tensor',
        # 'tsa_threshold': 'tsa_threshold_tensor'
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=metric_dict,
        every_n_iter=100
    )
    training_hooks = [logging_hook]

    global_step = tf.train.get_global_step()
    if warmup_steps > 0:
        warmup_lr = tf.to_float(global_step) / tf.to_float(warmup_steps) * lr
    else:
        warmup_lr = 0.0

    # decay the learning rate using the cosine schedule
    decay_lr = tf.train.cosine_decay(
        lr,
        global_step=global_step - warmup_steps,
        decay_steps=steps - warmup_steps,
        alpha=min_lr_ratio)

    learning_rate = tf.where(global_step < warmup_steps,
                             warmup_lr, decay_lr)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=0.9,
        use_nesterov=True)
    # grads_and_vars = optimizer.compute_gradients(total_loss)
    # gradients, variables = zip(*grads_and_vars)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train_op = optimizer.apply_gradients(
    #         zip(gradients, variables), global_step=tf.train.get_global_step())
    train_op = optimizer.minimize(
        total_loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode,
        loss=total_loss,
        training_hooks=training_hooks,
        train_op=train_op)


def input_fn(features, labels, batch_size, is_training=True, sup_only=False):
    if not is_training:
        features = features.astype(np.float32)
        labels = labels.astype(np.int32).reshape(len(labels))
        return tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)
    if sup_only:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        return dataset.shuffle(60000).repeat().batch(batch_size)
    sup = features['sup']
    unsup = features['unsup']
    aug = features['aug']

    sup = sup.astype(np.float32)
    unsup = unsup.astype(np.float32)
    aug = aug.astype(np.float32)
    labels = labels.astype(np.int64).reshape(len(labels))

    sup_x = tf.data.Dataset.from_tensor_slices(sup)
    sup_y = tf.data.Dataset.from_tensor_slices(labels)
    sup_dataset = tf.data.Dataset.zip((sup_x, sup_y))
    unsup = tf.data.Dataset.from_tensor_slices(unsup)
    aug = tf.data.Dataset.from_tensor_slices(aug)
    unsup_dataset = tf.data.Dataset.zip((unsup, aug))

    sup_dataset = sup_dataset.shuffle(sup_size).repeat().batch(
        batch_size,
        drop_remainder=True
    )
    unsup_dataset = unsup_dataset.shuffle(unsup_size).repeat().batch(
        batch_size * ratio,
        drop_remainder=True
    )
    sup_it = tf.compat.v1.data.make_one_shot_iterator(sup_dataset)
    unsup_it = tf.compat.v1.data.make_one_shot_iterator(unsup_dataset)

    return {
        'sup_dataset': sup_it.get_next(),
        'unsup_dataset': unsup_it.get_next()
    }


def main(argv):
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    run_config = tf.estimator.RunConfig().replace(session_config=session_config)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='ckpt',
        config=run_config,
        params={
            'n_classes': 10,
            'batch_size': batch_size,
            'sup_only': False
        })

    # classifier.train(
    #     input_fn=lambda: data.input_fn(batch_size=batch_size, ratio=ratio),
    #     steps=steps
    # )
    test_x = np.load('datasets/test_x.npy')
    test_y = np.load('datasets/test_y.npy')
    eval_result = classifier.evaluate(
        input_fn=lambda: input_fn(test_x, test_y, batch_size, is_training=False)
    )
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == "__main__":
    tf.app.run(main)
