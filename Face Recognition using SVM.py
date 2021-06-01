import face_recognition
from numpy import info
from sklearn import svm
import os
import cv2
from tkinter import *
from tkinter import messagebox
from tkinter import messagebox as mbox  

def img_capture():
    img_counter = 0
    # file_name = ''
    # ROOT = tk.Tk()
    # ROOT.withdraw()
    # file_name= simpledialog.askstring(title="Face Recognition",prompt="What's your Name?:")

    if(True):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Face Recognition")
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("Face Recognition", frame)

            k = cv2.waitKey(1)
            if k % 256 == 32:
                # SPACE pressed
                img_name = "test.jpg".format(img_counter)
                # path=os.path.normpath('C:\\Users\\Siddarth Saxena\\Desktop\\capture_test\\test')
                cv2.imwrite(os.path.join('test',img_name), frame)
                print("{} written!".format(img_name))
                print("Closing now")
                img_counter += 1
                break
    cam.release()

    cv2.destroyAllWindows()

    return "test" 

def display_name(list_name):
  window=Tk() 
  label = Label(window, text = "Faces Recognized") 
  listbox=Listbox(window,width=40)
  label.pack()
  listbox.pack(fill=BOTH, expand=1) #adds listbox to window
  [listbox.insert(END, row) for row in list_name] #one line for loop
  window.mainloop()  

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

train_dir = os.listdir('train/')

for person in train_dir:
    pix = os.listdir("train/"+ person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("train/"+ person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " was skipped and can't be used for training")

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

# Load the test image with unknown faces into a numpy array
name=img_capture()
test_image = face_recognition.load_image_file('test/'+name+'.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
num = len(face_locations)
print("Number of faces detected: ", num)

# Predict all the faces in the test image using the trained classifier
list_names=[]
print("Found:")
for i in range(num):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    list_names.append(*name)
display_name(list_names)