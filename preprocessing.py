from numpy import asarray
import numpy as np
import glob
import cv2
import cv

print(glob.glob("/Users/juhichecker/Desktop/SCU/Parallel Computing/project/test/*"))

l_files = glob.glob("/Users/juhichecker/Desktop/SCU/Parallel Computing/project/test/*")
print(len(l_files))

file = open("testP96.txt", "a+")
for l in l_files:
   #img = Image.open(l)
   print(l)
   img = cv2.imread(l)
   numpydata = asarray(img)
   print(numpydata.shape)
   print(len(numpydata))
   print(len(numpydata[0]))
   for i in range(len(numpydata)):
       for j in range(len(numpydata[0])):
            content = str(numpydata[i][j])
            content =content.strip('[')
            content = content.strip(']')
            file.write(content+"\n")
  
# # asarray() class is used to convert
# # PIL images into NumPy arrays
# numpydata = asarray(img)
  
# # <class 'numpy.ndarray'>
# print(type(numpydata))
  
# #  shape
# print(numpydata.shape)
