from constants import *

i = Image.open("92180.jpg")
i = pad_to_square(i)
i.save("92180_pad.jpg")
