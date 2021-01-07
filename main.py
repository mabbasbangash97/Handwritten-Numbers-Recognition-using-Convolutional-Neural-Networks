# @author Muhammad Abbas Bangash
#   051-18-0007

import tkinter
from PIL import ImageGrab
from PIL import Image
import cv2
import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import model_from_json

current_x, current_y = 0, 0
color = 'white'


def locate_xy(event):
    global current_x, current_y

    current_x, current_y = event.x, event.y


def drawline(event):
    global current_x, current_y

    canvas.create_line((current_x, current_y, event.x, event.y), fill=color, width=10)
    current_x, current_y = event.x, event.y


def show_color(new_color):
    global color
    color = new_color


def new():
    canvas.delete('all')
    colorbox()


def save():
    ImageGrab.grab().crop((60, 90, 1800, 990)).save('img.png')
    basewidth = 300
    baseheight = 400
    img = Image.open('img.png')
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save('img.png')
    hpercent = (baseheight / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((wsize, baseheight), Image.ANTIALIAS)
    img.save('img.png')
    img = cv2.imread('img.png')
    img_pil = Image.fromarray(img)
    img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
    #img_array = (img_28x28.flatten())
    img = img_28x28
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(gray, 2)
    print(img.shape)
    num_filters = 8
    filter_size = 3
    pool_size = 2

    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(10, activation='softmax'),
    ])
    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    from tensorflow.keras.utils import to_categorical

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    model.fit(
        mnist.train_images,
        to_categorical(mnist.train_labels),
        epochs=3,
        validation_data=(mnist.test_images, to_categorical(test_labels)),
    )

    model.load_weights('cnn.h5')


window = tkinter.Tk()

window.title('Number Detection By Muhammad Abbas')
window.state('zoomed')

window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)

menubar = tkinter.Menu(window)
window.config(menu=menubar)
submenu = tkinter.Menu(menubar, tearoff=0)
proj = tkinter.Menu(menubar, tearoff=0)

menubar.add_cascade(label='File', menu=submenu)
submenu.add_command(label='New', command=new)
submenu.add_command(label='Save', command=save)
menubar.add_cascade(label='Project By', menu=proj)
proj.add_command(label='Muhammad Abbas Bangash', command='')

canvas = tkinter.Canvas(window, background='black')
canvas.grid(row=0, column=0, sticky='nsew')

canvas.bind('<Button-1>', locate_xy)
canvas.bind('<B1-Motion>', drawline)


def colorbox():
    id1 = canvas.create_rectangle((10, 10, 30, 30), fill='white')
    canvas.tag_bind(id1, '<Button-1>', lambda x: show_color('white'))

    id2 = canvas.create_rectangle((40, 10, 60, 30), fill='gray')
    canvas.tag_bind(id2, '<Button-1>', lambda x: show_color('gray'))

    id3 = canvas.create_rectangle((70, 10, 90, 30), fill='brown4')
    canvas.tag_bind(id3, '<Button-1>', lambda x: show_color('brown4'))

    id4 = canvas.create_rectangle((100, 10, 120, 30), fill='red')
    canvas.tag_bind(id4, '<Button-1>', lambda x: show_color('red'))

    id5 = canvas.create_rectangle((130, 10, 150, 30), fill='orange')
    canvas.tag_bind(id5, '<Button-1>', lambda x: show_color('orange'))

    id6 = canvas.create_rectangle((160, 10, 180, 30), fill='yellow')
    canvas.tag_bind(id6, '<Button-1>', lambda x: show_color('yellow'))

    id7 = canvas.create_rectangle((190, 10, 210, 30), fill='green')
    canvas.tag_bind(id7, '<Button-1>', lambda x: show_color('green'))

    id8 = canvas.create_rectangle((220, 10, 240, 30), fill='blue')
    canvas.tag_bind(id8, '<Button-1>', lambda x: show_color('blue'))

    id9 = canvas.create_rectangle((250, 10, 270, 30), fill='purple')
    canvas.tag_bind(id9, '<Button-1>', lambda x: show_color('purple'))


colorbox()
window.mainloop()
