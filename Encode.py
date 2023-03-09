import cv2 as cv
import face_recognition
import pickle
import os

dataPath="data"
dataList=os.listdir(dataPath)
# print(dataList)
print("Getting Data...")
imgList=[]
imgIDs=[]
for img in dataList:
    # print(os.path.join(dataPath,img))
    imgList.append(cv.imread(os.path.join(dataPath,img)))
    # print(os.path.splitext(img))
    # print(os.path.splitext(img)[0])
    imgIDs.append(os.path.splitext(img)[0])

print("Data: ",imgIDs)

# making encode of images
def findEncodings(imgList):
    encodeList=[]
    for img in imgList:
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding started...")
knownEncodeList=findEncodings(imgList)
print("Encoding completed")

model=[knownEncodeList,imgIDs]
file=open("EncodeFile.p","wb")
pickle.dump(model,file)
file.close()
print("*** Model Saved ***")

