from keras.applications.vgg16 import VGG16
from quiver_engine import server
model = VGG16()

server.launch(model, input_folder='./samples_images/cat.png', temp_folder='./tmp', port=8000)
