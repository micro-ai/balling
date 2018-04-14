import os

for i, fn in enumerate(os.listdir('.')):
    if fn.startswith('.'):
        continue
    os.rename(fn, 'pingis-noplay-%02d.wav' % i)
