import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


def load_test(test_path, image_size, classes):
    X_test = [] 
    X_test_id = [] 

    for class_name in classes:
        path = os.path.join(test_path, class_name, '*g')
        files = sorted(glob.glob(path))

        # X_test = [] # Seriously?! Here!!??
        # X_test_id = [] # Not out side the loop??!!
        print("Reading test images")
        for fl in files:
            flbase = os.path.basename(fl)
            print(fl)
            img = cv2.imread(fl)
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            X_test.append(img)
            X_test_id.append(flbase)

  ### because we're not creating a DataSet object for the test images, normalization happens here
    X_test = np.array(X_test, dtype=np.uint8)
    X_test = X_test.astype('float32')
    X_test = X_test / 255.0

    return X_test, X_test_id



class DataSet(object):

  def __init__(self, images, labels, ids, cls):
    self._num_examples = images.shape[0]
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._ids = ids
    self._cls = cls
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def ids(self):
    return self._ids

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    assert batch_size <= self._num_examples
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch <= self._num_examples:
      end = self._index_in_epoch
      batch_images = self._images[start:end]
      batch_labels = self._labels[start:end]
      batch_ids = self._ids[start:end]
      batch_cls = self._cls[start:end]
    else:
      end1 = self._num_examples
      end2 = self._index_in_epoch - self._num_examples
      range1 = range(start, end1)
      range2 = range(0, end2)
      batch_images = self._images[range1 + range2]
      batch_labels = self._labels[range1 + range2]
      batch_ids = self._ids[range1 + range2]
      batch_cls = self._cls[range1 + range2] 
      # Finished epoch
      self._epochs_completed += 1
      self._index_in_epoch = end2

    return batch_images, batch_labels, batch_ids, batch_cls


def read_train_sets(train_path, image_size, classes, validation_size=0):
    class DataSets(object):
      pass
    data_sets = DataSets()

    images, labels, ids, cls = load_train(train_path, image_size, classes)
    images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

    if isinstance(validation_size, float):
      validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_ids = ids[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_ids = ids[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

    return data_sets


def read_test_set(test_path, image_size,classes):
    images, ids  = load_test(test_path, image_size,classes)
    return images, ids
