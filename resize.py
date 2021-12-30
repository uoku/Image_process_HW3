import os
from PIL import Image


imgs = os.listdir('pic4PR2')

for img in imgs:
    s = Image.open('pic4PR2/'+img)
    t = s.resize((512, 512))
    t.save('newimage/'+img)