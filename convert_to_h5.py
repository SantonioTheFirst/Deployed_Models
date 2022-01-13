from tensorflow.keras.models import load_model


cats = load_model('models/cats_vs_dogs')
cats.save('models/cats_vs_dogs.h5')

generator = load_model('models/generator')
generator.save('models/generator.h5')