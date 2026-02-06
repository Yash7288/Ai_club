#Learning image convolution
import math
import sys
from PIL import Image, ImageFilter

#Ensure correct usage
if len(sys.argv) != 2:
    sys.exit("Usage: python filter.py filename")

#Open image
image = Image.open(sys.argv[1]).convert("RGB")

#Filter image according to edge detection kernel
filtered = image.filter(ImageFilter.Kernel(
    size = (3,3),
    kernel = [-1,-1,-1,-1,8,-1,-1,-1,-1],
    scale = 1
))

#Show resulting image
filtered.show()

# import tensorflow as tf
# import sys
# import csv

# #Use MNIST handwriting dataset
# mnist = tf.keras.datasets.mnist

# #Prepare data for training
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train/255.0, x_test/255.0
# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)

# #Create a convolutional neural network
# model = tf.keras.models.Sequential([

#     #Convoluthional layer. Learn 32 filters using a 3x3 kernel
#     tf.keras.layers.Conv2D(
#         32, (3,3), activation = "relu", input_shape = (28, 28,1)
#     ),
    
#     #Max-pooling layer, using 2x2 pool size
#     tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

#     #Flatten units
#     tf.keras.layers.Flatten(),

#     #Add a hidden layer with dropout
#     tf.keras.layers.Dense(128, activation = "relu"),
#     tf.keras.layers.Dropout(0.5),

#     #Add an output layer units for all 10 digits
#     tf.keras.layers.Dense(10, activation = "softmax")
# ])

# model.compile(
#     optimizer = "adam",
#     loss = "categorical_crossentropy",
#     metrics = ["accuracy"]
# )

# model.fit(x_train, y_train, epochs = 10)

# #Evaluate neural network performace
# model.evaluate(x_test, y_test, verbose = 2)

# #save model to file
# if len(sys.argv) == 2:
#     filename = sys.argv[1]
#     model.save(filename)







# """Another thing"""
# # #Learning through CS50
# # with open("banknotes.csv") as f:
# #     reader = csv.reader(f)
# #     next(reader)

# #     data = []
# #     for row in reader:
# #         data.append({
# #             "evidence": [float(cell) for cell in row[:4]],
# #             "label": 1 if row[4] == "0" else 0
# #         })

# # #Separating data into training and testing groups
# # evidence = [row["evidence"] for row in data]
# # labels = [row["label"] for row in data]
# # X_training, X_testing, y_training, y_testing = train_test_split(evidence, labels, test_size = 0.4)

# # X_training = np.array(X_training, dtype=float)
# # X_testing  = np.array(X_testing, dtype=float)
# # y_training = np.array(y_training, dtype=float)
# # y_testing  = np.array(y_testing, dtype=float)


# # #Creating a nural network
# # model = tf.keras.models.Sequential()

# # #Add a hidden layer with 8 units, with ReLU activation
# # model.add(tf.keras.layers.Dense(8, input_shape = (4,), activation = "relu"))


# # #Adding a hidden layer with 1 unit, with sigmoid activation function
# # model.add(tf.keras.layers.Dense(1, activation="sigmoid"))


# # #Train neural network
# # model.compile(
# #     optimizer = "adam",
# #     loss = "binary_crossentropy",
# #     metrics = ["accuracy"]
# # )

# # model.fit(X_training, y_training, epochs = 24)

# # #Evaluate how well themodel performs
# # model.evaluate(X_testing, y_testing, verbose = 2)
