import numpy as np
import os
import csv
import cv2

label = []
count = 0

with open('test.csv', newline='') as csvFile:

    rows = csv.DictReader(csvFile)

    for row in rows:
        print('Dealing w/ '+row['filename'])
        print(row['ymin'], row['ymax'], row['xmin'], row['xmax'])

        # producing img label
        if row['label'] == 'good':
            label.append([1, 0, 0])
        elif row['label'] == 'bad':
            label.append([0, 1, 0])
        else:
            label.append([0, 0, 1])

        # load img
        if row['filename'] != '':
            img = cv2.imread(os.path.join('images', row['filename']))

            # resize img
            face_img = img[int(row['ymin'])-1:int(row['ymax']), int(row['xmin'])-1:int(row['xmax'])]
            resize_img = cv2.resize(face_img, (32, 32), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join('croppedImg_test', str(count)+'.jpg'), resize_img)
            count += 1

np.save('imgLabel_test',label)