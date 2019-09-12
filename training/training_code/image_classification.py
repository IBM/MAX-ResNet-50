import json
import os
import argparse
# keras layers
from keras.layers import Dense, GlobalAveragePooling2D
# keras applications
from keras.applications import ResNet50
from keras.applications.mobilenet import preprocess_input
# keras preprocessing
from keras.preprocessing.image import ImageDataGenerator
# keras optimizers
from keras.optimizers import Adam  # noqa
# keras functions
from keras.models import Model
from keras.backend import clear_session
#
import tensorflow as tf
from keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()

with open('param.json') as config_file:
    param_data = json.load(config_file)

data_dir = os.path.join(os.environ['DATA_DIR'], 'data')
result_dir = os.environ['RESULT_DIR']


def base_model_fn(model_name):
    return model_name(weights='imagenet', include_top=False)


def build_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D(name='Avg_pool_1')(x)
    x = Dense(1024, activation='relu', name='dense_one')(x)
    x = Dense(1024, activation='relu', name='dense_two')(x)
    x = Dense(512, activation='relu', name='dense_three')(x)
    x = Dense(int(args.output), activation='softmax', name='main_output')(x)
    return x


clear_session()
base_model = base_model_fn(ResNet50)
final_model = build_model(base_model)

model = Model(inputs=base_model.input, outputs=final_model)

for layer in base_model.layers:
    layer.trainable = False

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#
train_generator = train_datagen.flow_from_directory(data_dir,
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=param_data['batch_size'],
                                                    class_mode=param_data['class_mode'],
                                                    shuffle=True)

label_map = [[v, k] for k, v in train_generator.class_indices.items()]
label_dict = {}
for i in range(len(label_map)):
    label_dict[i] = label_map[i]

with open(os.path.join(result_dir, 'model/class_index.json'), 'w') as f:
    json.dump(label_dict, f)

# compile model
model.compile(optimizer=param_data['optimizer'],
              loss=param_data['loss'],
              metrics=list(param_data['metrics'].values()))
# calculate step size
step_size_train = train_generator.n // train_generator.batch_size
#
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=param_data['num_epochs'])

model.save(os.path.join(result_dir, 'model/resnet50.h5'))

builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(result_dir, 'model/tf'))

signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={'input': model.inputs[0]},
                                                                     outputs={'output': model.outputs[0]})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()
