# -*- coding: utf-8 -*-

import sys
import numpy as np
from PIL import Image

CLASS = 2

def main():
  
  img = Image.open(sys.argv[1])
  w, h = img.size
  target = np.asarray(img, dtype=np.int64)
  # print(target.shape)

  img = Image.open(sys.argv[2])
  pred = np.asarray(img, dtype=np.int64)[:,:,0]
  # print(pred.shape)

  acc = np.zeros(CLASS, dtype=np.int64)    # iをiと判定した数 acc[i]
  nacc = np.zeros(CLASS, dtype=np.int64)   # iをi以外と判定した数 nacc[i]
  occ = np.zeros(CLASS, dtype=np.int64)    # iに本来所属する画素数 occ[i]
  pcc = np.zeros(CLASS, dtype=np.int64)    # iに所属するとされた画素数 pcc[i]

  accuracy = np.zeros(CLASS, dtype=np.float64)
  iou = np.zeros(CLASS, dtype=np.float64)

  print(sys.argv[1])
  
  for y in range(0, h):
    for x in range(0, w):
      occ[target[y][x]] += 1
      pcc[pred[y][x]] += 1      
      
      if target[y][x] == pred[y][x]:
        acc[target[y][x]] += 1
      else:
        nacc[target[y][x]] += 1

  for i in range(0, CLASS):
    if acc[i] + nacc[i] > 0:
      print("Class {0:2d}: correct {1:6d}, incorrect {2:6d}, accuracy {3:.3}".format(i, acc[i], nacc[i], float(acc[i])/float(acc[i]+nacc[i])))
      accuracy[i] = float(acc[i])/float(acc[i]+nacc[i])
    else:
      print("Class {0:2d}: correct      0, incorrect      0, accuracy     -".format(i))
      accuracy[i] = -1

  print("Total   : correct {0:6d}, incorrect {1:6d}, accuracy {2:.3}".format(np.sum(acc), np.sum(nacc), float(np.sum(acc))/float(np.sum(acc)+np.sum(nacc))))

  cc = 0
  s = 0
  for a in accuracy:
    if a != -1:
      cc += 1
      s += a
  print("Mean Accuracy   : {0:.3}".format(s/float(cc)))
  print()

  for i in range(0, CLASS):
    if occ[i]+pcc[i]-acc[i] > 0:
      iou[i] = acc[i] / (occ[i]+pcc[i]-acc[i])
      print("Class {0:2d}: IoU {1:.3}".format(i, acc[i] / (occ[i]+pcc[i]-acc[i])))
    else:
      iou[i] = -1
      print("Class {0:2d}: IoU     -".format(i))
      
  print("Total IoU  : {0:.3}".format(np.sum(acc) / float(np.sum(occ)+np.sum(pcc)-np.sum(acc))))
  
  cc = 0
  s = 0
  for a in iou:
    if a != -1:
      cc += 1
      s += a
  print("Mean IoU   : {0:.3}".format(s/float(cc)))
  print()
  
if __name__ == "__main__":
  main()
