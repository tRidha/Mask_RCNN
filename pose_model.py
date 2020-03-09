import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import csv

class_names = []
rcnn_model = 0 

# --------------------------------------- MASK R CNN SETUP --------------------------------------- #
def init_mask_rcnn():
	global rcnn_model
	global class_names

	# Root directory of the project
	ROOT_DIR = os.path.abspath("/")

	# Import Mask RCNN
	sys.path.append(ROOT_DIR)  # To find local version of the library
	from mrcnn import utils
	import mrcnn.model as modellib
	from mrcnn import visualize
	# Import COCO config
	sys.path.append("samples/coco/")  # To find local version
	import COCO

	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Local path to trained weights file
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
	# Download COCO trained weights from Releases if needed
	if not os.path.exists(COCO_MODEL_PATH):
	    utils.download_trained_weights(COCO_MODEL_PATH)

	# Directory of images to run detection on
	IMAGE_DIR = os.path.join(ROOT_DIR, "images")

	class InferenceConfig(coco.CocoConfig):
	    # Set batch size to 1 since we'll be running inference on
	    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	    GPU_COUNT = 1
	    IMAGES_PER_GPU = 1

	config = InferenceConfig()

	# Create model object in inference mode.
	rcnn_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO
	rcnn_model.load_weights(COCO_MODEL_PATH, by_name=True)

	# COCO Class names
	# Index of the class in the list is its ID. For example, to get ID of
	# the teddy bear class, use: class_names.index('teddy bear')
	class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
	               'bus', 'train', 'truck', 'boat', 'traffic light',
	               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
	               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
	               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
	               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
	               'kite', 'baseball bat', 'baseball glove', 'skateboard',
	               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
	               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
	               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
	               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
	               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
	               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
	               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
	               'teddy bear', 'hair drier', 'toothbrush']

# ---------------------------------------- Helper functions for training ---------------------------------------- #

def load_Y_values(csv_filename):
  # Given a csv file, will return a Y matrix containing all the pose information
  # associated with each training example as well as a list of filenames
  with open(csv_filename, newline='') as csvfile:
    filenames = []
    reader = csv.reader(csvfile)
    data = list(reader)[1:]
    car_poses = []
    for i in range(len(data)):
      list_of_params = data[i][1].split()
      examples = []
      k = 0
      while k < len(list_of_params):
        examples.append(list_of_params[k+1:k+7])
        k += 7

      car_pose = min(examples,key=lambda x: x[5])
      car_poses.append(car_pose)
      filenames.append(str(data[i][0]) + '.jpg')

    Y = np.asarray(car_poses).T
    return Y, filenames

def extract_bounding_box_info(rcnn_model, filenames, show_images = False):
  # Given a list of images, runs each image through the trained rcnn_model
  # to output a corresponding list of bounding box information for the
  # car closest to the camera for each image.
  car_class_id = class_names.index('car')
  X = np.zeros((len(filenames),1028))

  for k in range(len(filenames)):
    filename = filenames[k]
    # Load image
    image = skimage.io.imread(IMAGE_PATH + filename)
    # if (os.path.exists(MASK_PATH + filename)):
    #   mask_image = skimage.io.imread(MASK_PATH + filename)
    #   mask = mask_image > 128
    #   image[mask] = 255
  
    # Run detection
    results = rcnn_model.detect([image], verbose=1)
    r = results[0]
    car_rois = []
    i = 0
    lowest_y = 0
    # Only process car images
    while i < len(r['class_ids']):
      if r['class_ids'][i] == car_class_id:
        
        y1,x1,y2,x2 = r['rois'][i]
        height = image.shape[0]
        width = image.shape[1]
        center_x = (x1 + x2) // 2
  
        # This is jank but it removes from consideration the car the camera is mounted on
        if not (y2 > height - 100 and center_x >= width * (1/3) and center_x <= width * (2/3)):
          roi = r['rois'][i]
          if roi[2] > lowest_y:
            lowest_y = roi[2]
            car_roi = roi
            feature_vec = r['features'][i].flatten()
      i += 1
    
    # For baseline, we will only consider cars that are closest to the camera (lowest-y value)
    print("Processed Image " + str(k))
    roi_and_features = np.concatenate([car_roi, feature_vec])
    X[k] = roi_and_features

  return X.T

# ---------------------------------------- Model Implementation ---------------------------------------- #
import tensorflow as tf
from tensorflow.python.framework import ops

# placeholders x y, forward prop z1, cost function mse loss = tf.reduce_mean(tf.squared_difference(prediction, Y)) 
# loss = tf.nn.l2_loss(prediction - Y),
# backprop tf.train.Adamoptimizer

def create_placeholders(n_x, n_y):

   
    X = tf.placeholder(dtype = tf.float32, shape=[n_x, None], name ='X')
    Y = tf.placeholder(dtype = tf.float32, shape=[n_y, None], name ='Y')
   
    
    return X, Y

def initialize_parameters():
    
    tf.set_random_seed(1)                   
    W1 = tf.get_variable("W1", [1024,1028], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [1024,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [1024,1024], initializer = tf.contrib.layers.xavier_initializer(seed = 2))
    b2 = tf.get_variable("b2", [1024,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6,1024], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_propagation(X, parameters):
     
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
                  
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.tanh(Z1)   
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                                            
                                                
    return Z3

def compute_cost(Z3, Y, threshold = 2.8):
  
    t_hat = Z3[3:6]
    t = Y[3:6]

    huber_loss = tf.keras.losses.Huber(delta=threshold)
    t_cost = huber_loss(t, t_hat)
    #  tf.cond(tf.norm(t - t_hat) < threshold, lambda: tf.squared_difference(t, t_hat), lambda : tf.norm(t - t_hat) - (0.5 * threshold))

    r_hat = Z3[:3]
    r = Y[:3]

    r_cost = tf.squared_difference(r, r_hat)

    cost = tf.reduce_mean(t_cost + r_cost)
    #cost = tf.reduce_mean(tf.nn.cross_entropy_with_logits(logits = logits, labels = labels))
    
    return cost

def pose_model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,
          num_epochs = 1000, print_cost = True):
    
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
  
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
  
    init = tf.global_variables_initializer()

  
    with tf.Session() as sess:
        
        
        sess.run(init)
        
        
        for epoch in range(num_epochs):

            epoch_cost = 0.                       
             
            seed = seed + 1
            

            #for minibatch in minibatches:

                #(minibatch_X, minibatch_Y) = minibatch
                
            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

                
                #epoch_cost += minibatch_cost / minibatch_size

            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch >= 1000 and epoch % 5 == 0:
                costs.append(minibatch_cost)
                
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")



        # correct_prediction = tf.equal(tf.argmax(Z1), tf.argmax(Y))

        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


        correct_rot = tf.cast(tf.less(tf.squared_difference(Z3[:3], Y[:3]), [1]), "float")
        correct_trans = tf.cast(tf.less(tf.squared_difference(Z3[3:], Y[3:]), [2.7]), "float")
        accuracy = tf.reduce_mean(tf.cast(correct_trans * correct_rot, "float"))

        print ("Train Mean Squared Difference:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Train:", Z3[5].eval({X: X_train, Y: Y_train}))
        print(Y_train[5])
        print ("Test Mean Squared Difference:", accuracy.eval({X: X_test, Y: Y_test}))
        print ("Test:", Z3[5].eval({X: X_test, Y: Y_test}))
        print(Y_test[5])
        
        return parameters

def main():
	args = sys.argv[1:]

	init_mask_rcnn()

	if len(args) == 4:
		if args[0] == '-preprocess':

			train_file = args[1]
			test_file = args[2]
			out_file = args[3]
			# Construct training set
			Y_train, tr_filenames = load_Y_values(train_file)
			X_train = extract_bounding_box_info(rcnn_model, tr_filenames)

			# Construct test set
			Y_test, test_filenames = load_Y_values(test_file)
			X_test = extract_bounding_box_info(rcnn_model, test_filenames)

			np.savetxt(out_file + '_ytrain.csv', Y_train, delimiter = ',')
			np.savetxt(out_file + '_xtrain.csv', X_train, delimiter = ',')
			np.savetxt(out_file + '_ytest.csv', Y_test, delimiter = ',')
			np.savetxt(out_file + '_xtest.csv', X_test, delimiter = ',')





if __name__ == "__main__": 
	main()