import cv2
# Specify the Path location of the file to use.
video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("dataset/haarcascade_smile.xml")

while True:
    # open webcam.
    success, img = video.read()
    # convert the image to gray image .
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # read faces using an already included haarcascade file and detectMultiscale().
    faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)
    cnt = 500
    # waitKey(1) will display a frame for 1 ms.
    # after which display will be automatically closed.
    keyPressed = cv2.waitKey(1)
    
    for x, y, w, h in faces:
        # draw an outer boundary of the face using rectangle().
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 3)
        # read smiles using an already included haarcascade file and detectMultiscale().
        smiles = smileCascade.detectMultiScale(grayImg, 1.8, 15)
        
        for x, y, w, h in smiles:
            # draw an outer boundary of the smile using rectangle().
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (100, 100, 100), 5)
            print("Image "+str(cnt)+"Saved")
            # provide the location of the folder in which we want to save the images.
            path=r'result_photo\photo'+str(cnt)+'.jpg'
            # save the images.
            cv2.imwrite(path, img)
            cnt += 1
            # we will just save 3 images in one run.
            # thus useif statement which breaks the loop if var >= 3.
            if(cnt >= 503):
                break

    # Show output.
    cv2.imshow('live video', img)
    # for break loop
    if(keyPressed & 0xFF == ord('q')):
        break

video.release()
cv2.destroyAllWindows()
