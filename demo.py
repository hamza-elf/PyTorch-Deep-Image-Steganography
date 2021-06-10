import argparse
import os
import shutil
import socket
import time


import utils.transformed as transforms
from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet
import PySimpleGUI as sg
from pathlib import Path
from PIL import Image
from io import BytesIO
import os
import shutil

os.mkdir('test')


def update_image(image_element, filename, savename):
    im = Image.open(filename)
    im.save('test/' + savename)
    w, h = size_of_image
    scale = max(im.width / w, im.height / h)
    im = im.resize((int(im.width / scale), int(im.height / scale)), resample=Image.CUBIC)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    image_element.update(data=data)


def process_images():
    os.system('python main.py --test=./test')


# sg.theme('DarkAmber')   # Add a touch of color
w, h = size_of_image = (250, 250)
# All the stuff inside your window.
                  [sg.Image(key='-IMCOUV-', size=(250, 250))],
                  [sg.FileBrowse('Secret'), sg.InputText(enable_events=True, key='-FSEC-')],
                  [sg.Image(key='-IMSEC-', size=(250, 250))],
                  [sg.Button('Procéder', enable_events=True, key='-PROCESS-',visible=False),
                   sg.Button('Cancel')]]
layout_decoder = [[sg.Text('Réseau Décodeur')],
                  [sg.Text('Image Container')],
layout_text = [[sg.Image(filename='ensias.png', pad=((0,0),(0,450)))],
               {sg.Text('Encadré par : \nPr.Abdelaziz S.DOUKKALI', justification='left', font=("Helvetica", 12),
                        text_color='black')}]
layout_decoder = [[sg.Text('Réseau Décodeur', font=("Helvetica", 25))],
                  [sg.Text('Image Stego', font=("Helvetica", 16))],
                  [sg.Image(key='-IMCONTAINER-', size=(250, 250)),
                   sg.Button('Extraire', enable_events=True, key='-EXTRACT-',visible=False)],
                  [sg.Text('Image Secrète extraite')],
                   sg.Button('Extraire', font=("Helvetica", 12), button_color=('','green'), enable_events=True, key='-EXTRACT-', visible=False)],
                  [sg.Text('Image Secrète extraite',font=("Helvetica", 16))],
                  [sg.Image(key='-IMREVSEC-', size=(250, 250))]]
layout = [[sg.Column(layout_encoder, pad=(0, 0)),
           sg.VSep(),
           sg.Column(layout_decoder, pad=(0, 0))]]

# Create the Window
window = sg.Window('Stéganographie par CNN', layout, size=(900, 600), icon='loupe.ico')
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
        shutil.rmtree('test', ignore_errors=True)
        break
    if event == '-FCOUV-':
        update_image(window['-IMCOUV-'], values['-FCOUV-'], savename="couverture.jpg")
    elif event == '-FSEC-':
        update_image(window['-IMSEC-'], values['-FSEC-'], savename="secret.jpg")
        window['-PROCESS-'].update(visible=True)
    if event == '-PROCESS-':
        process_images()
        path = str(Path('test'))
        window['-IMCONTAINER-'].update(filename=path + '/Container.png')
        window['-EXTRACT-'].update(visible=True)
    elif event == '-EXTRACT-':
        window['-IMREVSEC-'].update(filename=path + '/RevSec.png')
window.close()