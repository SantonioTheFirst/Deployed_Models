from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import Address

# from tensorflow.keras.models import load_model

# import numpy as np
# from numpy import expand_dims, asarray, round
# from numpy.random import normal

# from matplotlib import pyplot as plt

# from PIL import Image
# import io

# np.random.seed(1000)
# gan = load_model('models/result_gan_nontrainable.h5')
# generator = load_model('models/generator')
# latent_dims = 100
# number_of_photos = 25

# cats_dogs_classifier = load_model('models/cats_vs_dogs.h5')


app = FastAPI(docs_url=None, redoc_url=None)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')


# def plot_images(images):
#     figure = plt.figure(figsize=(5, 5))
#     print(images.shape)
#     for i in range(number_of_photos):
#         plt.subplot(5, 5, i + 1)
#         plt.imshow((images[i] + 1) / 2)
#         plt.axis('off')
#         plt.tight_layout()
#     plt.savefig('static/generated.png')


def prediction_result(prediction):
    return f'Cat probability: {round(prediction[0][0] * 100, 4)}%, \
         Dog probability: {round(prediction[0][1] * 100, 4)}%'


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    # print(request)
    return templates.TemplateResponse('index.html', {'request': request})


# @app.get('/cats_vs_dogs', response_class=HTMLResponse)
# def cats_vs_dogs(request: Request):
#     return templates.TemplateResponse(
#         'cats_vs_dogs.html', {
#             'request': request})


# @app.post('/cats_vs_dogs', response_class=HTMLResponse)
# async def cats_vs_dogs(request: Request, file_input: UploadFile = File(...)):
#     file_format = file_input.content_type
#     if 'image' in file_format:
#         file_data = await file_input.read()
#         im = Image.open(io.BytesIO(file_data))
#         im_arr = asarray(asarray(im))
#         filename = 'image.jpg'
#         im.save(f'static/{filename}')
#         im.close()
#         prediction = prediction_result(
#             cats_dogs_classifier.predict(
#                 expand_dims(
#                     im_arr, axis=0)))
#         image = True
#     else:
#         prediction = 'Select an image file!'
#         image = False
#     return templates.TemplateResponse(
#         'cats_vs_dogs.html', {
#             'request': request, 'prediction': prediction, 'image': image})


# @app.get('/ganime', response_class=HTMLResponse)
# def ganime(request: Request):
#     return templates.TemplateResponse('ganime.html', {'request': request})


# @app.post('/ganime', response_class=HTMLResponse)
# def ganime(request: Request):
#     # prediction = gan.layers[0](np.random.normal(
#     #     0, 1, (number_of_photos, latent_dims)))

#     prediction = generator.predict(normal(0, 1, (number_of_photos, latent_dims)))

#     plot_images(prediction)
#     return templates.TemplateResponse(
#         'ganime.html', {'request': request, 'img': True})
