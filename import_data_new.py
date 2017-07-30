# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 20:11:45 2017

@author: Administrator
"""

import h5py
import numpy

class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
            "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
            self._num_examples = images.shape[0]
            #print (0)
            self._images = images
            self._labels = labels
            self._epochs_completed = 0
            self._index_in_epoch = 0
    
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
          fake_image = [1.0 for _ in range(784)]
          fake_label = 0
          return [fake_image for _ in range(batch_size)], [fake_label for _ in range(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        #print (0)
        #print(self._index_in_epoch,self._num_examples)
        #若当前训练读取的index>总体的images数时，则读取读取开始的batch_size大小的数据
        if self._index_in_epoch > self._num_examples:
            #print (0)
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        #print ("start is:%d,end is:%d"%(start,end))
        return self._images[start:end], self._labels[start:end]


def read_data_sets():
  class DataSets(object):
    pass
  matfn=r'C:\Users\Administrator.PC-20170605CXBN\py_files\logistic regression\L944_F-4.mat'
  odata=h5py.File(matfn)
  data_sets = DataSets()
  train_datas = odata['train_x']
  train_labels = odata['train_y']
  print ('the shape of train sets:',train_datas.shape,train_labels.shape)
  test_datas = odata['test_x']
  test_labels = odata['test_y']
  print ('the shape of train sets:',test_datas.shape,test_labels.shape)
  data_sets.train = DataSet(train_datas, train_labels)
  data_sets.test = DataSet(test_datas, test_labels)
  return data_sets,odata