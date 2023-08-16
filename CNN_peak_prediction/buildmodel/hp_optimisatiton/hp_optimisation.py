import tensorflow as tf
from tensorboard.plugins.hparams import api as hp



#buildmodel - uneeded
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#definemodel
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'


#buildmodel - at top + in mid where graphsa are made
from tensorflow.summary import FileWriter

file_writer = FileWriter('./logs/hparam_tuning')
summary = tf.compat.v1.Summary()

file_writer.add_summary(summary)


#definemodel
hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
)


#2
#definemodel - bottom
def train_test_model(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  ])
  
  #this part
  model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )

  #buildmodel - end of each training epoch/gen
  model.fit(x_train, y_train, epochs=5) # Run with 1 epoch to speed things up for demo purposes
  _, accuracy = model.evaluate(x_test, y_test)
  return accuracy
  #return accuracy
  
  




#buildmodel - end of each training epoch/gen
def run(run_dir, hparams):
    with tf.compat.v1.summary.FileWriter(run_dir) as writer:  # Use tf.compat.v1.summary.FileWriter
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        
        # Create a summary manually
        summary = tf.compat.v1.Summary()
        summary.value.add(tag=METRIC_ACCURACY, simple_value=accuracy)
        
        # Write the summary to the file
        writer.add_summary(summary, global_step=1)
        

#buildmodel - end of each training epoch/gen
session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      #run_name = "run-%d" % session_num
      run_name = "run-{}".format({h.name: hparams[h] for h in hparams})
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1


print("[INFO] for model summary: \n")
print("tensorboard dev upload --logdir logs/hparam_tuning \\")
print("--name \" OPTIONAL \" \\")
print("--description \" OPTIONAL \" ")
print("\n")

