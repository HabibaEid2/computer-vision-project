import cv2
import os 
import numpy as np
import customtkinter as ctk
from PIL import Image , ImageTk

#components

main_color1 = '#001BB7'
main_color2 = '#FAF3E1'


# create the main page of the user

def showMainPage() : 

    main_color1 = '#222222'

    mainPage = ctk.CTk()
    mainPage.geometry('1000x600+200+20')
    mainPage._set_appearance_mode('light')
    mainPage.title('Face Detection And Recognition App')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    imagesPath = os.listdir('users')
    usersNames = []
    facesList = []

    for img in imagesPath : 
        usersNames.append(img[:img.index('.')])
        img = cv2.cvtColor(cv2.imread(f'users/{img}'), cv2.COLOR_BGR2GRAY)
        imgFaces = face_cascade.detectMultiScale(img , 1.3 , 5)
        (x , y , w , h) = imgFaces[0]
        face = img[y:y+h+50 , x:x+w] # cut the image
        face = cv2.resize(face , (200 , 200))
        facesList.append(face)


    def openLiveCamera() : 
        cap = cv2.VideoCapture(0)
        while True : 
            ret , frame = cap.read()
            gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
            facesFounded = face_cascade.detectMultiScale(gray_frame , 1.3 , 5)

            for i in range(len(facesFounded)) : 
                (x , y , w , h) = facesFounded[i]

                face = gray_frame[y:y+h+50 , x:x+w]
                face = cv2.resize(face , (200 , 200))

                model = cv2.face.LBPHFaceRecognizer.create()
                indexes = []
                for i in range(len(imagesPath)) : 
                    indexes.append(i)
                model.train(facesList , np.array(indexes))

                label , confidence = model.predict(face)

                if confidence < 60 : 
                    cv2.putText(frame, usersNames[label], (x , y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 , 255 , 0) , 1, cv2.LINE_AA)
                    cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 255 , 0) , 2)
                else : 
                    cv2.putText(frame, 'Unknown', (x , y- 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 , 0 , 255) , 1, cv2.LINE_AA)
                    cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 0 , 2500) , 2)

            cv2.imshow('my video' , frame)
            if cv2.waitKey(1) == ord('q') : 
                break 

        cap.release()
        cv2.destroyAllWindows()

    def getResults( resultsContainer, user) : 
        for frame in resultsContainer.winfo_children() :
            frame.destroy()

        if user == 'all' : 
            for i in range(len(imagesPath)) : 
                userImage = Image.open(f'./users/{imagesPath[i]}')
                userImage = userImage.resize((300 , 350))
                userImage = ImageTk.PhotoImage(userImage)
                userFrame = ctk.CTkFrame(
                    resultsContainer , 
                    width = 200 , 
                    height= 250 , 
                    corner_radius=10 , 
                    fg_color='white'
                )
                userFrame.grid(row = i // 3 , column = i % 3 , padx = 7 , pady = 7)
                userNameLabel = ctk.CTkLabel(userFrame , text=usersNames[i] , font  = ('Times New Roman' , 16 , 'bold') , text_color='#666')
                userNameLabel.place(x = 3 , y = 3)
                userImageLabel = ctk.CTkLabel(userFrame , text= '' , image=userImage)
                userImageLabel.place(x = 0 , y = 40)
        elif user in usersNames : 
            userImage = Image.open(f'./users/{user}.png')
            userImage = userImage.resize((450 , 500))
            userImage = ImageTk.PhotoImage(userImage)
            userFrame = ctk.CTkFrame(
                resultsContainer , 
                width = 300 , 
                height= 350 , 
                corner_radius=10 ,
                fg_color='white'
            )
            userFrame.place(x = 150 , y = 0)
            userNameLabel = ctk.CTkLabel(userFrame , text=user , font  = ('Times New Roman' , 16 , 'bold') , text_color='#666')
            userNameLabel.place(x = 3 , y = 3)
            userImageLabel = ctk.CTkLabel(userFrame , text= '' , image=userImage)
            userImageLabel.place(x = 0 , y = 40)

    def addUser() : 
        mainPage.destroy()
        registerPage()

    # start sidebar
    sideFrame = ctk.CTkFrame(mainPage , fg_color= main_color1 , corner_radius=0 , width = 50 , height=600)
    sideFrame.place(x = 0 , y = 0)

    cameraImg = Image.open('./gui_imgs/play.png')
    cameraImg = cameraImg.resize((50 , 50))
    cameraImg = ImageTk.PhotoImage(cameraImg)

    cameraButtom = ctk.CTkButton(
        sideFrame , 
        text= '' , 
        width = 20 , 
        fg_color= main_color1 , 
        image= cameraImg , 
        cursor = 'hand2' , 
        command= openLiveCamera
    )
    cameraButtom.place(x = 0 , y = 5)

    addImg = Image.open('./gui_imgs/add.png')
    addImg = addImg.resize((45 , 45))
    addImg = ImageTk.PhotoImage(addImg)

    addNewUserButton = ctk.CTkButton(
        sideFrame , 
        text= '' , 
        width = 20 , 
        fg_color= main_color1 , 
        image= addImg , 
        cursor = 'hand2' , 
        command= addUser
    )
    addNewUserButton.place(x = 0 , y = 550)

    # end sidebar

    # start container

    containerFrame = ctk.CTkFrame(mainPage , fg_color=main_color2 , corner_radius=0 , width = (1000-50) , height=600)
    containerFrame.place(x = 50 , y = 0)

    selectAllButton = ctk.CTkButton(
        containerFrame , 
        text = 'All',
        text_color='white' , 
        cursor = 'hand2' ,
        width = 70 , 
        height=37 ,
        fg_color=main_color1 , 
        corner_radius=10 , 
        command= lambda : getResults(resultsContainer , 'all')
    )
    selectAllButton.place(x = 10 , y = 20)

    searchBar = ctk.CTkEntry(
        containerFrame ,
        width = 600 ,
        height= 37 , 
        placeholder_text='search with name' ,
        placeholder_text_color= '#999' , 
        corner_radius= 10
    )
    searchBar._set_appearance_mode('light')
    searchBar.place(x = 85 , y = 20)

    searchButton = ctk.CTkButton(
        containerFrame ,
        text = 'search',
        text_color='white' , 
        cursor = 'hand2' ,
        width = 70 , 
        height=37 ,
        fg_color=main_color1 , 
        corner_radius=10 , 
        command= lambda : getResults(resultsContainer , searchBar.get())
    )
    searchButton.place(x = 700 , y = 20)

    txtLabel = ctk.CTkLabel(containerFrame , text='founded results' , font=('Times New Roman' , 17 , 'normal') , text_color='#777')
    txtLabel.place(x = 350 , y = 80)

    global resultsContainer 
    resultsContainer = ctk.CTkScrollableFrame(
        containerFrame , 
        width = 770 , 
        height=(600-110) , 
        fg_color = main_color2  , 
    )
    resultsContainer.place(x = 150 , y = 110)


    mainPage.mainloop()

# create register page 
def registerPage() : 
    registerPage = ctk.CTk()
    registerPage.geometry('1000x600+200+20')
    registerPage._set_appearance_mode('light')
    registerPage.title('Face Detection And Recognition App')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    # Image Section 

    imageFrame = ctk.CTkFrame(
        registerPage ,
        width = 500 ,
        height=600
    )
    imageFrame.place(x = 0 , y = 0)

    bgImg = Image.open('./gui_imgs/face-scan.png')
    bgImg = bgImg.resize((800 , 900))
    bgImg = ImageTk.PhotoImage(bgImg)

    imgLabel = ctk.CTkLabel(
        imageFrame , 
        text= '' , 
        image= bgImg
    )
    imgLabel.place(x = 0 , y = 0)

    # Register Section
    registerFrame = ctk.CTkFrame(
        registerPage , 
        width=500 , 
        height= 600 , 
        fg_color= main_color2 , 
        corner_radius= 0
    )
    registerFrame.place(x = 500 , y = 0)

    regardLabel = ctk.CTkLabel(
        registerFrame , 
        text= 'WELCOME TO OUR FACE IDENTIFICATION AND RECOGNITION APP. PLEASE LOG IN TO CONTINUE...' , 
        text_color= main_color1 , 
        font= ('Times New Roman' , 20 , 'bold'), 
        wraplength= 400
    )
    regardLabel.place(x = 65 , y = 150)

    cameraImg = Image.open('./gui_imgs/photo-camera.png')
    cameraImg = cameraImg.resize((90 , 90))
    cameraImg = ImageTk.PhotoImage(cameraImg)

    def showCamera() : 
        cap = cv2.VideoCapture(0)
        while True : 
            ret , frame = cap.read()
            cv2.imshow('camera' , frame)
            if cv2.waitKey(1) == ord('k') :
                global userImage 
                userImage = frame
                break
            elif cv2.waitKey(1) == ord('q') : 
                break 
        cap.release()
        cv2.destroyAllWindows()

    cameraButtom = ctk.CTkButton(
        registerFrame , 
        text= '' , 
        fg_color= main_color2 , 
        hover_color=main_color2 , 
        image= cameraImg , 
        cursor = 'hand2' , 
        command= showCamera 
    )
    cameraButtom.place(x = 185 , y = 310)

    def register() : 
        userNameInputValue = userNameInput.get()
        cv2.imwrite(f'./users/{userNameInputValue}.png' , userImage)
        registerPage.destroy()
        showMainPage()

    userNameInput = ctk.CTkEntry(
        registerFrame , 
        width= 300 , 
        height=30 , 
        border_color= '#aaa' , 
        border_width= 1 , 
        placeholder_text='Enter your name...' , 
        placeholder_text_color= '#999'
    )
    userNameInput._set_appearance_mode('light')
    userNameInput.place(x = 115 , y = 380)

    registerButton = ctk.CTkButton(
        registerFrame , 
        text='register' , 
        text_color= main_color2 ,
        fg_color=main_color1 , 
        width= 70 , 
        corner_radius= 5 , 
        cursor = 'hand2' , 
        command= register
    )
    registerButton.place(x = 225 , y = 430)

    registerPage.mainloop()

registerPage()