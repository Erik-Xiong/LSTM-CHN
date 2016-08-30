
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import math
import os
import csv
import pickle
import numpy as np

import tensorflow as tf 
from tensorflow.contrib import learn

DATAPATH = "addr_sample.csv"
# CHAR_TO_DELETE = {ord(c): None for c in ' 【】!"'}
MAX_DOC_LENGTH = 50

def tokenizer(iterator):
  for addr in iterator:
    yield list(addr)
  # for value in iterator:
  #   ss = []
  #   for v in value:
  #     ss.append(v)
  #   yield ss
def process_data(path):
  with open(path, newline='', encoding='utf-8') as f:
    reader=csv.reader(f)
    X = []
    for line in reader:
      line = line[0].strip()
      if not line:
        continue
      # line = line.translate(None,' 【】!"')
      origin = list(line)
      if len(origin) >= 50:
        origin = origin[:49]
      X.append(origin + ["<EOS/>"])
  return np.array(X)

def iter_docs(docs):
  for doc in docs:
    n_parts = int(math.ceil(float(len(doc)) / MAX_DOC_LENGTH))
    for part in range(n_parts):
      offset_begin = part * MAX_DOC_LENGTH
      offset_end = offset_begin + MAX_DOC_LENGTH
      inp = np.zeros(MAX_DOC_LENGTH, dtype=np.int32)
      out = np.zeros(MAX_DOC_LENGTH, dtype=np.int32)
      inp[:min(offset_end - offset_begin, len(doc) - offset_begin)] = \
        doc[offset_begin:offset_end]
      out[:min(offset_end - offset_begin, len(doc) - offset_begin - 1)] = \
        doc[offset_begin + 1: offset_end + 1]
      yield inp, out

def unzip_xy(iterable):
  X, y = itertools.tee(iterable)
  return (item[0] for item in X), (item[1] for item in y)
  # return list(zip(*iterable))

vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOC_LENGTH, min_frequency=2, tokenizer_fn=tokenizer)
datao = process_data(DATAPATH)
vocab_processor.fit(datao)

f = open("vocab.txt", "w")
for k, v in vocab_processor.vocabulary_._mapping.items():
  f.write("%s\t%d\n" % (k.encode('utf8'), v))
f.close()
n_words = len(vocab_processor.vocabulary_)
print("total workds:" + str(n_words))

HIDDEN_SIZE = 874

def get_language_model(hidden_size):
  def language_model(X, y):
    inputs = learn.ops.one_hot_matrix(X, n_words)
    inputs = tf.unpack(inputs, axis=1)
    target = tf.unpack(y, axis=1)
    encoder_cell = tf.nn.rnn_cell.OutputProjectionWrapper(tf.nn.rnn_cell.GRUCell(hidden_size), n_words)
    output, _ = tf.nn.rnn(encoder_cell, inputs, dtype=tf.float32)
    return learn.ops.sequence_classifier(output, target)
  return language_model

def exp_decay(global_step):
  return tf.train.exponential_decay(0.001, global_step, 5000, 0.5, staircase = True)


model_path = 'address_logs'
if os.path.exists(model_path):
  estimator = learn.TensorFlowEstimator.restore(model_path)
else:
  estimator = learn.TensorFlowEstimator(model_fn=get_language_model(HIDDEN_SIZE), n_classes=n_words, optimizer="Adam", learning_rate=exp_decay, steps = 16273, batch_size=64, continue_training=True)
# while True:
#   try:
perm = np.random.permutation(len(datao))
datao = datao[perm]
data = vocab_processor.transform(datao)
X, y = unzip_xy(iter_docs(data))
estimator.fit(X, y)

  #   estimator.save(model_path)
  # except KeyboardInterrupt:
    # estimator.save(model_path)
    # break
a = "北京东城区内环到三环里北京市东城区和平里七区乙16号楼113室"
b = '啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊'
print(estimator.predict(vocab_processor.transform(a)).sum()/float(len(a)))
print(estimator.predict(vocab_processor.transform(b)).sum()/float(len(b)))