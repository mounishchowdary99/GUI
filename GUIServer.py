from flask import Flask, request, render_template, redirect, url_for, session
import pypyodbc

import json
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.io import imread
from os.path import basename
from glob import glob
import string
import random
import os
import sys
import hashlib
import shutil

from keras.models import Model
from keras.layers import LSTM, Bidirectional, Dense, Input, Embedding, concatenate, Dropout
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from PIL import Image as PImage
from keras.models import Sequential
from keras.layers import RepeatVector, Input, Dropout, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, UpSampling2D, Reshape, Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.backend import clear_session
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras import *


from RoleModel import RoleModel
from UserModel import UserModel
from Constants import connString


app = Flask(__name__)
app.secret_key = "MySecret"
ctx = app.app_context()
ctx.push()

with ctx:
    pass

userName = ""
roleObject = None
message = ""
msgType = ""

def initialize():
    global message, msgType
    message = ""
    msgType=""

def processRole(optionID):
    
    if optionID == 10 :
        if roleObject.canRole == False :
            return False
    if optionID == 20 :
        if roleObject.canUser == False :
            return False
    if optionID == 30 :
        if roleObject.CL111 == False :
            return False
    if optionID == 40 :
        if roleObject.CL222 == False :
            return False
    if optionID == 50 :
        if roleObject.CL333 == False :
            return False
    return True

@app.route('/')
def index():
    global userID, userName
    return render_template('Login.html')  # when the home page is called Index.hrml will be triggered.

@app.route('/processLogin', methods=['POST'])
def processLogin():
    global userID, userName, roleObject
    userName= request.form['userName']
    password= request.form['password']
    conn1 = pypyodbc.connect(connString, autocommit=True)
    cur1 = conn1.cursor()
    sqlcmd1 = "SELECT * FROM UserTable WHERE userName = '"+userName+"' AND password = '"+password+"' AND isActive = 1"; 
    
    cur1.execute(sqlcmd1)
    row = cur1.fetchone()
    
    cur1.commit()
    if not row:
        return render_template('Login.html', processResult="Invalid Credentials")
    userID = row[0]
    userName = row[3]
    
    cur2 = conn1.cursor()
    sqlcmd2 = "SELECT * FROM Role WHERE RoleID = '"+str(row[6])+"'"; 
    cur2.execute(sqlcmd2)
    row2 = cur2.fetchone()
   
    if not row2:
        return render_template('Login.html', processResult="Invalid Role")
    
    roleObject = RoleModel(row2[0], row2[1],row2[2],row2[3],row2[4],row2[5])

    return render_template('Dashboard.html')

@app.route("/ChangePassword")
def changePassword():
    global userID, userName
    return render_template('ChangePassword.html')

@app.route("/ProcessChangePassword", methods=['POST'])
def processChangePassword():
    global userID, userName
    oldPassword= request.form['oldPassword']
    newPassword= request.form['newPassword']
    confirmPassword= request.form['confirmPassword']
    conn1 = pypyodbc.connect(connString, autocommit=True)
    cur1 = conn1.cursor()
    sqlcmd1 = "SELECT * FROM UserTable WHERE userName = '"+userName+"' AND password = '"+oldPassword+"'"; 
    cur1.execute(sqlcmd1)
    row = cur1.fetchone()
    cur1.commit()
    if not row:
        return render_template('ChangePassword.html', msg="Invalid Old Password")
    
    if newPassword.strip() != confirmPassword.strip() :
       return render_template('ChangePassword.html', msg="New Password and Confirm Password are NOT same")
    
    conn2 = pypyodbc.connect(connString, autocommit=True)
    cur2 = conn2.cursor()
    sqlcmd2 = "UPDATE UserTable SET password = '"+newPassword+"' WHERE userName = '"+userName+"'"; 
    cur1.execute(sqlcmd2)
    cur2.commit()
    return render_template('ChangePassword.html', msg="Password Changed Successfully")


@app.route("/Dashboard")
def Dashboard():
    global userID, userName
    return render_template('Dashboard.html')


@app.route("/Information")
def Information():
    global message, msgType
    return render_template('Information.html', msgType=msgType, message = message)




@app.route("/UserListing")

def UserListing():
    global userID, userName
    
    global message, msgType, roleObject
    if roleObject == None:
        message = "Application Error Occurred"
        msgType="Error"
        return redirect(url_for('Information'))
    canRole = processRole(10)

    if canRole == False:
        message = "You Don't Have Permission to Access User"
        msgType="Error"
        return redirect(url_for('Information'))
    
    conn2 = pypyodbc.connect(connString, autocommit=True)
    cursor = conn2.cursor()
    sqlcmd1 = "SELECT * FROM UserTable ORDER BY userName"
    cursor.execute(sqlcmd1)
    records = []
    
    while True:
        dbrow = cursor.fetchone()
        if not dbrow:
            break
        
        conn3 = pypyodbc.connect(connString, autocommit=True)
        cursor3 = conn3.cursor()
        temp = str(dbrow[6])
        sqlcmd3 = "SELECT * FROM Role WHERE RoleID = '"+temp+"'"
        cursor3.execute(sqlcmd3)
        rolerow = cursor3.fetchone()
        roleModel = RoleModel(0)
        if rolerow:
           roleModel = RoleModel(rolerow[0],rolerow[1])
        else:
           print("Role Row is Not Available")
        
        row = UserModel(dbrow[0], dbrow[1], dbrow[2], dbrow[3], dbrow[4], dbrow[5], dbrow[6], roleModel=roleModel)
        records.append(row)
    return render_template('UserListing.html', records=records)


@app.route("/UserOperation")
def UserOperation():
    
    global userID, userName
    
    global message, msgType, roleObject
    if roleObject == None:
        message = "Application Error Occurred"
        msgType="Error"
        return redirect(url_for('Information'))
    canRole = processRole(10)

    if canRole == False:
        message = "You Don't Have Permission to Access User"
        msgType="Error"
        return redirect(url_for('Information'))
    
    operation = request.args.get('operation')
    unqid = ""
    
    
    
    rolesDDList = []
    
    conn4 = pypyodbc.connect(connString, autocommit=True)
    cursor4 = conn4.cursor()
    sqlcmd4 = "SELECT * FROM Role"
    cursor4.execute(sqlcmd4)
    print("sqlcmd4???????????????????????????????????????????????????????/", sqlcmd4)
    while True:
        roleDDrow = cursor4.fetchone()
        if not roleDDrow:
            break
        print("roleDDrow[1]>>>>>>>>>>>>>>>>>>>>>>>>>", roleDDrow[1])
        roleDDObj = RoleModel(roleDDrow[0], roleDDrow[1])
        rolesDDList.append(roleDDObj)
        
        
    row = UserModel(0)

    if operation != "Create" :
        unqid = request.args.get('unqid').strip()
        conn2 = pypyodbc.connect(connString, autocommit=True)
        cursor = conn2.cursor()
        sqlcmd1 = "SELECT * FROM UserTable WHERE UserID = '"+unqid+"'"
        cursor.execute(sqlcmd1)
        dbrow = cursor.fetchone()
        if dbrow:
            
            conn3 = pypyodbc.connect(connString, autocommit=True)
            cursor3 = conn3.cursor()
            temp = str(dbrow[6])
            sqlcmd3 = "SELECT * FROM Role WHERE RoleID = '"+temp+"'"
            cursor3.execute(sqlcmd3)
            rolerow = cursor3.fetchone()
            roleModel = RoleModel(0)
            if rolerow:
               roleModel = RoleModel(rolerow[0],rolerow[1])
            else:
               print("Role Row is Not Available")
            row = UserModel(dbrow[0], dbrow[1], dbrow[2], dbrow[3], dbrow[4], dbrow[5], dbrow[6], roleModel=roleModel)
        
    return render_template('UserOperation.html', row = row, operation=operation, rolesDDList=rolesDDList )




@app.route("/ProcessUserOperation",methods = ['POST'])
def processUserOperation():
    global userName, userID
    operation = request.form['operation']
    unqid = request.form['unqid'].strip()
    userName= request.form['userName']
    emailid= request.form['emailid']
    password=request.form['password']
    contactNo= request.form['contactNo']
    isActive = 0
    if request.form.get("isActive") != None :
        isActive = 1
    roleID= request.form['roleID']
    
    
    conn1 = pypyodbc.connect(connString, autocommit=True)
    cur1 = conn1.cursor()
    
    
    if operation == "Create" :
        sqlcmd = "INSERT INTO UserTable( userName,emailid, password,contactNo, isActive, roleID) VALUES('"+userName+"','"+emailid+"', '"+password+"' , '"+contactNo+"', '"+str(isActive)+"', '"+str(roleID)+"')"
    if operation == "Edit" :
        sqlcmd = "UPDATE UserTable SET userName = '"+userName+"', emailid = '"+emailid+"', password = '"+password+"',contactNo='"+contactNo+"',  isActive = '"+str(isActive)+"', roleID = '"+str(roleID)+"' WHERE UserID = '"+unqid+"'"  
    if operation == "Delete" :

        sqlcmd = "DELETE FROM UserTable WHERE UserID = '"+unqid+"'" 

    if sqlcmd == "" :
        return redirect(url_for('Information')) 
    cur1.execute(sqlcmd)
    cur1.commit()
    conn1.close()
    return redirect(url_for("UserListing"))







'''
    Role Operation Start
'''

@app.route("/RoleListing")
def RoleListing():
    
    global message, msgType
    print("roleObject>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", roleObject)
    if roleObject == None:
        message = "Application Error Occurred"
        msgType="Error"
        return redirect(url_for('Information'))
    canRole = processRole(20)

    if canRole == False:
        message = "You Don't Have Permission to Access Role"
        msgType="Error"
        return redirect(url_for('Information'))
    
    searchData = request.args.get('searchData')
    print(searchData)
    if searchData == None:
        searchData = "";
    conn2 = pypyodbc.connect(connString, autocommit=True)
    cursor = conn2.cursor()
    sqlcmd1 = "SELECT * FROM Role WHERE roleName LIKE '"+searchData+"%'"
    print(sqlcmd1)
    cursor.execute(sqlcmd1)
    records = []
    
    while True:
        dbrow = cursor.fetchone()
        if not dbrow:
            break
        
        row = RoleModel(dbrow[0],dbrow[1],dbrow[2],dbrow[3],dbrow[4],dbrow[5],dbrow[6])
        
        records.append(row)
    
    return render_template('RoleListing.html', records=records, searchData=searchData)

@app.route("/RoleOperation")
def RoleOperation():
    
    global message, msgType
    if roleObject == None:
        message = "Application Error Occurred"
        msgType="Error"
        return redirect(url_for('/'))
    canRole = processRole(120)

    if canRole == False:
        message = "You Don't Have Permission to Access Role"
        msgType="Error"
        return redirect(url_for('Information'))
    
    operation = request.args.get('operation')
    unqid = ""
    row = RoleModel(0, "",0,0,0,0)
    if operation != "Create" :
        unqid = request.args.get('unqid').strip()
        
        
        conn2 = pypyodbc.connect(connString, autocommit=True)
        cursor = conn2.cursor()
        sqlcmd1 = "SELECT * FROM Role WHERE RoleID = '"+unqid+"'"
        cursor.execute(sqlcmd1)
        while True:
            dbrow = cursor.fetchone()
            if not dbrow:
                break
            row = RoleModel(dbrow[0],dbrow[1],dbrow[2],dbrow[3],dbrow[4],dbrow[5],dbrow[6])
        
    return render_template('RoleOperation.html', row = row, operation=operation )


@app.route("/ProcessRoleOperation", methods=['POST'])
def ProcessRoleOperation():
    global message, msgType
    if roleObject == None:
        message = "Application Error Occurred"
        msgType="Error"
        return redirect(url_for('/'))
    canRole = processRole(120)

    if canRole == False:
        message = "You Don't Have Permission to Access Role"
        msgType="Error"
        return redirect(url_for('Information'))
    
    
    print("ProcessRole")
    
    operation = request.form['operation']
    if operation != "Delete" :
        roleName = request.form['roleName']
        canRole = 0
        canUser = 0
        CL111 = 0
        CL222 = 0
        CL333 = 0
        
        
        
        if request.form.get("canRole") != None :
            canRole = 1
        if request.form.get("canUser") != None :
            canUser = 1
        if request.form.get("CL111") != None :
            CL111 = 1
        if request.form.get("CL222") != None :
            CL222 = 1
        if request.form.get("CL333") != None :
            CL333 = 1
        
        
    
    print(1)
    unqid = request.form['unqid'].strip()
    print(operation)
    conn3 = pypyodbc.connect(connString, autocommit=True)
    cur3 = conn3.cursor()
    
    
    sqlcmd = ""
    if operation == "Create" :
        sqlcmd = "INSERT INTO Role (roleName, canRole, canUser, CL111, CL222, CL333) VALUES ('"+roleName+"', '"+str(canRole)+"', '"+str(canUser)+"', '"+str(CL111)+"', '"+str(CL222)+"', '"+str(CL333)+"')"
    if operation == "Edit" :
        print("edit inside")
        sqlcmd = "UPDATE Role SET roleName = '"+roleName+"', canRole = '"+str(canRole)+"', canUser = '"+str(canUser)+"', CL111 = '"+str(CL111)+"', CL222 = '"+str(CL222)+"', CL333 = '"+str(CL333)+"' WHERE RoleID = '"+unqid+"'" 
    if operation == "Delete" :
        conn4 = pypyodbc.connect(connString, autocommit=True)
        cur4 = conn4.cursor()
        sqlcmd4 = "SELECT roleID FROM UserTable WHERE roleID = '"+unqid+"'" 
        cur4.execute(sqlcmd4)
        dbrow4 = cur4.fetchone()
        if dbrow4:
            message = "You can't Delete this Role Since it Available in Users Table"
            msgType="Error"
            return redirect(url_for('Information')) 
        
        sqlcmd = "DELETE FROM Role WHERE RoleID = '"+unqid+"'" 
    print(operation, sqlcmd)
    if sqlcmd == "" :
        return redirect(url_for('Information')) 
    cur3.execute(sqlcmd)
    cur3.commit()
    
    return redirect(url_for('RoleListing')) 
    
'''
    Role Operation End
'''




@app.route("/First50FilesInfo")
def FilesInfo():
    global message, msgType


    return render_template('First50FilesInfo.html')

@app.route("/First50FilesInfoResult", methods=["POST"])
def get_files_info_result():
    ds_dir = os.path.join('static', 'datasets', "web", "all_data")
    print(ds_dir)
    img_files = glob(os.path.join(ds_dir, '*', '*'))
    df = pd.DataFrame(dict(path=[x for x in img_files if x.endswith('png') or x.endswith('gui')]))
    print(df)
    df['source'] = df['path'].map(lambda x: x.split('\\')[-2])
    df['filetype'] = df['path'].map(lambda x: os.path.splitext(x)[1][1:])
    df['fileid'] = df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
    records = df.head(50).values.tolist()
    return render_template('First50FilesInfoResult.html', records=records)


@app.route("/Last50FilesInfo")
def last_50_files_info():
    global message, msgType


    return render_template('Last50FilesInfo.html')

@app.route("/Last50FilesInfoResult", methods=["POST"])
def last_50_files_info_result():
    ds_dir = os.path.join('static', 'datasets', "web", "all_data")
    img_files = glob(os.path.join(ds_dir, '*', '*'))
    df = pd.DataFrame(dict(path=[x for x in img_files if x.endswith('png') or x.endswith('gui')]))
    df['source'] = df['path'].map(lambda x: x.split('\\')[-2])
    df['filetype'] = df['path'].map(lambda x: os.path.splitext(x)[1][1:])
    df['fileid'] = df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
    records = df.tail(50).values.tolist()
    return render_template('Last50FilesInfoResult.html', records=records)


@app.route("/Random50FilesInfo")
def random_50_files_info():
    global message, msgType


    return render_template('Random50FilesInfo.html')

@app.route("/Random50FilesInfoResult", methods=["POST"])
def random_50_files_info_result():
    ds_dir = os.path.join('static', 'datasets', "web", "all_data")
    img_files = glob(os.path.join(ds_dir, '*', '*'))
    df = pd.DataFrame(dict(path=[x for x in img_files if x.endswith('png') or x.endswith('gui')]))
    df['source'] = df['path'].map(lambda x: x.split('\\')[-2])
    df['filetype'] = df['path'].map(lambda x: os.path.splitext(x)[1][1:])
    df['fileid'] = df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
    records = df.tail(50).values.tolist()
    return render_template('Random50FilesInfoResult.html', records=records)
@app.route("/ConvertToColumn")
def convert_to_column():
    global message, msgType
    return render_template('ConvertToColumn.html')
data_df = pd.DataFrame()
@app.route("/ConvertToColumnResult", methods=["POST"])
def convert_to_column_result():
    global data_df
    ds_dir = os.path.join('static', 'datasets')
    img_files = glob(os.path.join(ds_dir, '*', '*'))
    df = pd.DataFrame(dict(path=[x for x in img_files if x.endswith('png') or x.endswith('gui')]))
    df['source'] = df['path'].map(lambda x: x.split('\\')[-2])
    df['filetype'] = df['path'].map(lambda x: os.path.splitext(x)[1][1:])
    df['fileid'] = df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
    data_df = df.pivot_table(index=['source', 'fileid'],
                                     columns=['filetype'],
                                     values='path',
                                     aggfunc='first').reset_index()

    return render_template('ConvertToColumnResult.html')



@app.route("/First50FilesInfoAfterConversion")
def first_50_files_info_after_conversion():
    global message, msgType


    return render_template('First50FilesInfoAfterConversion.html')

@app.route("/First50FilesInfoAfterConversionResult", methods=["POST"])
def first_50_files_info_after_conversion_result():
    global data_df, msgType, message

    if len(data_df) == 0:
        message = "First Convert To Column"
        msgType = "Error"
        return redirect(url_for('Information'))
    print(data_df.head(3))
    records = data_df.head(50).values.tolist()
    return render_template('First50FilesInfoAfterConversionResult.html', records=records)


@app.route("/Last50FilesInfoAfterConversion")
def last_50_files_info_after_conversion():
    global message, msgType
    return render_template('Last50FilesInfoAfterConversion.html')

@app.route("/Last50FilesInfoAfterConversionResult", methods=["POST"])
def Last_50_files_info_after_conversion_result():
    global data_df, msgType, message

    if len(data_df) == 0:
        message = "First Convert To Column"
        msgType = "Error"
        return redirect(url_for('Information'))
    records = data_df.tail(50).values.tolist()
    return render_template('Last50FilesInfoAfterConversionResult.html', records=records)



@app.route("/Random50FilesInfoAfterConversion")
def random_50_files_info_after_conversion():
    global message, msgType


    return render_template('Random50FilesInfoAfterConversion.html')

@app.route("/Random50FilesInfoAfterConversionResult", methods=["POST"])
def random_50_files_info_after_conversion_result():
    global data_df, msgType, message

    if len(data_df) == 0:
        message = "First Convert To Column"
        msgType = "Error"
        return redirect(url_for('Information'))
    records = data_df.sample(50).values.tolist()
    return render_template('Random50FilesInfoAfterConversionResult.html', records=records)

@app.route("/FindAvailableSamples")
def find_available_samples():
    global message, msgType
    return render_template('FindAvailableSamples.html')

@app.route("/FindAvailableSamplesResult", methods=["POST"])
def find_available_samples_result():
    global data_df, msgType, message

    if len(data_df) == 0:
        message = "First Convert To Column"
        msgType = "Error"
        return redirect(url_for('Information'))

    cnt = data_df.shape[0]
    return render_template('FindAvailableSamplesResult.html', cnt=cnt)

def read_text_file(in_path):
        with open(in_path, 'r') as f:
            return f.read()

def imread_scale(in_path):
        return np.array(PImage.open(in_path).resize((64, 64), PImage.ANTIALIAS))[:, :, :3]

@app.route("/ShowImages")
def show_images():
    global data_df, msgType, message

    if len(data_df) == 0:
        message = "First Convert To Column"
        msgType = "Error"
        return redirect(url_for('Information'))
    return render_template('ShowImages.html')
@app.route("/ShowImagesResult", methods=["POST"])
def show_images_result():
    global data_df, msgType, message

    if len(data_df) == 0:
        message = "First Convert To Column"
        msgType = "Error"
        return redirect(url_for('Information'))


    clear_sample_df = data_df.groupby(['source']).apply(lambda x: x.sample(1)).reset_index(drop=True)

    fig, m_axs = plt.subplots(3, clear_sample_df.shape[0], figsize=(12, 12))
    for (_, c_row), (im_ax, im_lr, gui_ax) in zip(clear_sample_df.iterrows(), m_axs.T):
        im_ax.imshow(imread(c_row['png']))
        im_ax.axis('off')
        im_ax.set_title(c_row['source'])

        im_lr.imshow(imread_scale(c_row['png']), interpolation='none')
        im_lr.set_title('LowRes')

        gui_ax.text(0, 0, read_text_file(c_row['gui']),
                    style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.1, 'pad': 10},
                    fontsize=7)
        gui_ax.axis('off')
    return render_template('ShowImagesResult.html')


@app.route("/ConstructDataSet")
def construct_dataset():
    return render_template('ConstructDataSet.html')

@app.route("/ConstructDataSetProcess", methods=["POST"])
def construct_dataset_process():
    print("construct_dataset_process Called")
    TRAINING_SET_NAME = "training"
    EVALUATION_SET_NAME = "eval"
    input_path = "static/datasets/web/all_data"
    paths = []
    distribution = 6
    for f in os.listdir(input_path):
        if f.find(".gui") != -1:
            path_gui = "{}/{}".format(input_path, f)
            file_name = f[:f.find(".gui")]
            print(file_name)
            if os.path.isfile("{}/{}.png".format(input_path, file_name)):
                path_img = "{}/{}.png".format(input_path, file_name)
                paths.append(file_name)

    evaluation_samples_number = len(paths) / (distribution + 1)
    training_samples_number = evaluation_samples_number * distribution

    assert training_samples_number + evaluation_samples_number == len(paths)

    print("Splitting datasets, training samples: {}, evaluation samples: {}".format(training_samples_number,
                                                                                    evaluation_samples_number))

    np.random.shuffle(paths)

    eval_set = []
    train_set = []

    hashes = []
    for path in paths:
        if sys.version_info >= (3,):
            f = open("{}/{}.gui".format(input_path, path), 'r', encoding='utf-8')
        else:
            f = open("{}/{}.gui".format(input_path, path), 'r')

        with f:
            chars = ""
            for line in f:
                chars += line
            content_hash = chars.replace(" ", "").replace("\n", "")
            content_hash = hashlib.sha256(content_hash.encode('utf-8')).hexdigest()

            if len(eval_set) == evaluation_samples_number:
                train_set.append(path)
            else:
                is_unique = True
                for h in hashes:
                    if h is content_hash:
                        is_unique = False
                        break

                if is_unique:
                    eval_set.append(path)
                else:
                    train_set.append(path)

            hashes.append(content_hash)

    assert len(eval_set) == evaluation_samples_number
    assert len(train_set) == training_samples_number

    if not os.path.exists("{}/{}".format(os.path.dirname(input_path), EVALUATION_SET_NAME)):
        os.makedirs("{}/{}".format(os.path.dirname(input_path), EVALUATION_SET_NAME))

    if not os.path.exists("{}/{}".format(os.path.dirname(input_path), TRAINING_SET_NAME)):
        os.makedirs("{}/{}".format(os.path.dirname(input_path), TRAINING_SET_NAME))

    for path in eval_set:
        shutil.copyfile("{}/{}.png".format(input_path, path),
                        "{}/{}/{}.png".format(os.path.dirname(input_path), EVALUATION_SET_NAME, path))
        shutil.copyfile("{}/{}.gui".format(input_path, path),
                        "{}/{}/{}.gui".format(os.path.dirname(input_path), EVALUATION_SET_NAME, path))

    for path in train_set:
        shutil.copyfile("{}/{}.png".format(input_path, path),
                        "{}/{}/{}.png".format(os.path.dirname(input_path), TRAINING_SET_NAME, path))
        shutil.copyfile("{}/{}.gui".format(input_path, path),
                        "{}/{}/{}.gui".format(os.path.dirname(input_path), TRAINING_SET_NAME, path))

    print("Training dataset: {}/training_set".format(os.path.dirname(input_path), path))
    print("Evaluation dataset: {}/eval_set".format(os.path.dirname(input_path), path))
    return render_template('ConstructDataSetProcess.html')


class Utils:
    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1

            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)

    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")

CONTEXT_LENGTH = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 10
STEPS_PER_EPOCH = 72000

@app.route("/ConvertImageToArray")
def convert_image_to_numpy_array():
    return render_template('ConvertImageToArray.html')

@app.route("/ConvertImageToArrayProcess", methods=["POST"])
def convert_image_to_numpy_array_process():
    print("convert_image_to_numpy_array_process Called")

    input_path = "static/datasets/web/all_data"
    output_path = "static/datasets/web/all_data/training_features"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for f in os.listdir(input_path):
        if f.find(".png") != -1:
            img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)
            file_name = f[:f.find(".png")]

            np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
            retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

            assert np.array_equal(img, retrieve)

            shutil.copyfile("{}/{}.gui".format(input_path, file_name), "{}/{}.gui".format(output_path, file_name))

    print("Numpy arrays saved in {}".format(output_path))

    return render_template('ConvertImageToArrayProcess.html')


START_TOKEN = "<START>"
END_TOKEN = "<END>"
PLACEHOLDER = " "
SEPARATOR = '->'

class AModel:
    def __init__(self, input_shape, output_size, output_path):
        self.model = None
        self.input_shape = input_shape
        self.output_size = output_size
        self.output_path = output_path
        self.name = ""

    def save(self):
        model_json = self.model.to_json()
        with open("{}/{}.json".format(self.output_path, self.name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("{}/{}.h5".format(self.output_path, self.name))

    def load(self, name=""):
        output_name = self.name if name == "" else name
        with open("{}/{}.json".format(self.output_path, output_name), "r") as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("{}/{}.h5".format(self.output_path, output_name))

class Vocabulary:
    def __init__(self):
        self.binary_vocabulary = {}
        self.vocabulary = {}
        self.token_lookup = {}
        self.size = 0

        self.append(START_TOKEN)
        self.append(END_TOKEN)
        self.append(PLACEHOLDER)

    def append(self, token):
        if token not in self.vocabulary:
            self.vocabulary[token] = self.size
            self.token_lookup[self.size] = token
            self.size += 1

    def create_binary_representation(self):
        if sys.version_info >= (3,):
            items = self.vocabulary.items()
        else:
            items = self.vocabulary.iteritems()
        for key, value in items:
            binary = np.zeros(self.size)
            binary[value] = 1
            self.binary_vocabulary[key] = binary

    def get_serialized_binary_representation(self):
        if len(self.binary_vocabulary) == 0:
            self.create_binary_representation()

        string = ""
        if sys.version_info >= (3,):
            items = self.binary_vocabulary.items()
        else:
            items = self.binary_vocabulary.iteritems()
        for key, value in items:
            array_as_string = np.array2string(value, separator=',', max_line_width=self.size * self.size)
            string += "{}{}{}\n".format(key, SEPARATOR, array_as_string[1:len(array_as_string) - 1])
        return string

    def save(self, path):
        output_file_name = "{}/words.vocab".format(path)
        output_file = open(output_file_name, 'w')
        output_file.write(self.get_serialized_binary_representation())
        output_file.close()

    def retrieve(self, path):
        input_file = open("{}/words.vocab".format(path), 'r')
        buffer = ""
        for line in input_file:
            try:
                separator_position = len(buffer) + line.index(SEPARATOR)
                buffer += line
                key = buffer[:separator_position]
                value = buffer[separator_position + len(SEPARATOR):]
                value = np.fromstring(value, sep=',')

                self.binary_vocabulary[key] = value
                self.vocabulary[key] = np.where(value == 1)[0][0]
                self.token_lookup[np.where(value == 1)[0][0]] = key

                buffer = ""
            except ValueError:
                buffer += line
        input_file.close()
        self.size = len(self.vocabulary)

class Dataset:
    def __init__(self):
        self.input_shape = None
        self.output_size = None

        self.ids = []
        self.input_images = []
        self.partial_sequences = []
        self.next_words = []

        self.voc = Vocabulary()
        self.size = 0

    @staticmethod
    def load_paths_only(path):
        print("Parsing data...")
        gui_paths = []
        img_paths = []
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                path_gui = "{}/{}".format(path, f)
                gui_paths.append(path_gui)
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    path_img = "{}/{}.png".format(path, file_name)
                    img_paths.append(path_img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    path_img = "{}/{}.npz".format(path, file_name)
                    img_paths.append(path_img)

        assert len(gui_paths) == len(img_paths)
        return gui_paths, img_paths

    def load(self, path, generate_binary_sequences=False):
        print("Loading data...")
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                gui = open("{}/{}".format(path, f), 'r')
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    img = Utils.get_preprocessed_img("{}/{}.png".format(path, file_name), IMAGE_SIZE)
                    self.append(file_name, gui, img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    img = np.load("{}/{}.npz".format(path, file_name))["features"]
                    self.append(file_name, gui, img)

        print("Generating sparse vectors...")
        self.voc.create_binary_representation()
        self.next_words = self.sparsify_labels(self.next_words, self.voc)
        if generate_binary_sequences:
            self.partial_sequences = self.binarize(self.partial_sequences, self.voc)
        else:
            self.partial_sequences = self.indexify(self.partial_sequences, self.voc)

        self.size = len(self.ids)
        assert self.size == len(self.input_images) == len(self.partial_sequences) == len(self.next_words)
        assert self.voc.size == len(self.voc.vocabulary)

        print("Dataset size: {}".format(self.size))
        print("Vocabulary size: {}".format(self.voc.size))

        self.input_shape = self.input_images[0].shape
        self.output_size = self.voc.size

        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

    def convert_arrays(self):
        print("Convert arrays...")
        self.input_images = np.array(self.input_images)
        self.partial_sequences = np.array(self.partial_sequences)
        self.next_words = np.array(self.next_words)

    def append(self, sample_id, gui, img, to_show=False):
        if to_show:
            pic = img * 255
            pic = np.array(pic, dtype=np.uint8)
            Utils.show(pic)

        token_sequence = [START_TOKEN]
        for line in gui:
            line = line.replace(",", " ,").replace("\n", " \n")
            tokens = line.split(" ")
            for token in tokens:
                self.voc.append(token)
                token_sequence.append(token)
        token_sequence.append(END_TOKEN)

        suffix = [PLACEHOLDER] * CONTEXT_LENGTH

        a = np.concatenate([suffix, token_sequence])
        for j in range(0, len(a) - CONTEXT_LENGTH):
            context = a[j:j + CONTEXT_LENGTH]
            label = a[j + CONTEXT_LENGTH]

            self.ids.append(sample_id)
            self.input_images.append(img)
            self.partial_sequences.append(context)
            self.next_words.append(label)

    @staticmethod
    def indexify(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def binarize(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.binary_vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def sparsify_labels(next_words, voc):
        temp = []
        for label in next_words:
            temp.append(voc.binary_vocabulary[label])

        return temp

    def save_metadata(self, path):
        np.save("{}/meta_dataset".format(path), np.array([self.input_shape, self.output_size, self.size]))

class Generator:
    @staticmethod
    def data_generator(voc, gui_paths, img_paths, batch_size, input_shape, generate_binary_sequences=False, verbose=False, loop_only_one=False, images_only=False):
        assert len(gui_paths) == len(img_paths)
        voc.create_binary_representation()

        while 1:
            batch_input_images = []
            batch_partial_sequences = []
            batch_next_words = []
            sample_in_batch_counter = 0

            for i in range(0, len(gui_paths)):
                if img_paths[i].find(".png") != -1:
                    img = Utils.get_preprocessed_img(img_paths[i], IMAGE_SIZE)
                else:
                    img = np.load(img_paths[i])["features"]
                gui = open(gui_paths[i], 'r')

                token_sequence = [START_TOKEN]
                for line in gui:
                    line = line.replace(",", " ,").replace("\n", " \n")
                    tokens = line.split(" ")
                    for token in tokens:
                        voc.append(token)
                        token_sequence.append(token)
                token_sequence.append(END_TOKEN)

                suffix = [PLACEHOLDER] * CONTEXT_LENGTH

                a = np.concatenate([suffix, token_sequence])
                for j in range(0, len(a) - CONTEXT_LENGTH):
                    context = a[j:j + CONTEXT_LENGTH]
                    label = a[j + CONTEXT_LENGTH]

                    batch_input_images.append(img)
                    batch_partial_sequences.append(context)
                    batch_next_words.append(label)
                    sample_in_batch_counter += 1

                    if sample_in_batch_counter == batch_size or (loop_only_one and i == len(gui_paths) - 1):
                        if verbose:
                            print("Generating sparse vectors...")
                        batch_next_words = Dataset.sparsify_labels(batch_next_words, voc)
                        if generate_binary_sequences:
                            batch_partial_sequences = Dataset.binarize(batch_partial_sequences, voc)
                        else:
                            batch_partial_sequences = Dataset.indexify(batch_partial_sequences, voc)

                        if verbose:
                            print("Convert arrays...")
                        batch_input_images = np.array(batch_input_images)
                        batch_partial_sequences = np.array(batch_partial_sequences)
                        batch_next_words = np.array(batch_next_words)

                        if verbose:
                            print("Yield batch")
						#include a generator for images only for autoencoder
                        if images_only:
                            yield(batch_input_images, batch_input_images)
                        else:
                            yield ([batch_input_images, batch_partial_sequences], batch_next_words)

                        batch_input_images = []
                        batch_partial_sequences = []
                        batch_next_words = []
                        sample_in_batch_counter = 0


class autoencoder_image(AModel):
	def __init__(self, input_shape, output_size, output_path):
		AModel.__init__(self, input_shape, output_size, output_path)
		self.name = 'autoencoder'

		input_image = Input(shape=input_shape)
		encoder = Conv2D(32, 3, padding='same', activation='relu')(input_image)
		encoder = Conv2D(32, 3, padding='same', activation='relu')(encoder)
		encoder = MaxPooling2D()(encoder)
		encoder = Dropout(0.25)(encoder)

		encoder = Conv2D(64, 3, padding='same', activation='relu')(encoder)
		encoder = Conv2D(64, 3, padding='same', activation='relu')(encoder)
		encoder = MaxPooling2D()(encoder)
		encoder = Dropout(0.25)(encoder)

		encoder = Conv2D(128, 3, padding='same', activation='relu')(encoder)
		encoder = Conv2D(128, 3, padding='same', activation='relu')(encoder)
		encoder = MaxPooling2D()(encoder)
		encoded = Dropout(0.25, name='encoded_layer')(encoder)

		decoder = Conv2DTranspose(128, 3, padding='same', activation='relu')(encoded)
		decoder = Conv2DTranspose(128, 3, padding='same', activation='relu')(decoder)
		decoder = UpSampling2D()(decoder)
		decoder = Dropout(0.25)(decoder)

		decoder = Conv2DTranspose(64, 3, padding='same', activation='relu')(decoder)
		decoder = Conv2DTranspose(64, 3, padding='same', activation='relu')(decoder)
		decoder = UpSampling2D()(decoder)
		decoder = Dropout(0.25)(decoder)

		decoder = Conv2DTranspose(32, 3, padding='same', activation='relu')(decoder)
		decoder = Conv2DTranspose(3, 3, padding='same', activation='relu')(decoder)
		decoder = UpSampling2D()(decoder)
		decoded = Dropout(0.25)(decoder)

		# decoder = Dense(256*256*3)(decoder)
		# decoded = Reshape(target_shape=input_shape)(decoder)

		self.model = Model(input_image, decoded)
		self.model.compile(optimizer='adadelta', loss='binary_crossentropy')
		self.model.summary()

	def fit_generator(self, generator, steps_per_epoch):
		self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1)
		self.save()

	def predict_hidden(self, images):
		hidden_layer_model = Model(inputs = self.input, outputs = self.get_layer('encoded_layer').output)
		return hidden_layer_model.predict(images)


class image_to_code(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "image_to_code"

        visual_input = Input(shape=input_shape)

        # Load the pre-trained autoencoder model
        autoencoder_model = autoencoder_image(input_shape, input_shape, output_path)
        autoencoder_model.load('autoencoder')
        autoencoder_model.model.load_weights('static/model/autoencoder.h5')

        # Get only the model up to the encoded part
        hidden_layer_model_freeze = Model(inputs=autoencoder_model.model.input,
                                          outputs=autoencoder_model.model.get_layer('encoded_layer').output)
        hidden_layer_input = hidden_layer_model_freeze(visual_input)

        # Additional layers before concatenation
        hidden_layer_model = Flatten()(hidden_layer_input)
        hidden_layer_model = Dense(1024, activation='relu')(hidden_layer_model)
        hidden_layer_model = Dropout(0.3)(hidden_layer_model)
        hidden_layer_model = Dense(1024, activation='relu')(hidden_layer_model)
        hidden_layer_model = Dropout(0.3)(hidden_layer_model)
        hidden_layer_result = RepeatVector(CONTEXT_LENGTH)(hidden_layer_model)

        # Make sure the loaded hidden_layer_model_freeze will no longer be updated
        for layer in hidden_layer_model_freeze.layers:
            layer.trainable = False


        language_model = Sequential()
        language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(LSTM(128, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        decoder = concatenate([hidden_layer_result, encoded_text])

        decoder = LSTM(512, return_sequences=True)(decoder)
        decoder = LSTM(512, return_sequences=False)(decoder)
        decoder = Dense(output_size, activation='softmax')(decoder)

        self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)

        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def fit_generator(self, generator, steps_per_epoch):
        self.model.summary()
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1)
        self.save()

    def predict(self, image, partial_caption):
        return self.model.predict([image, partial_caption], verbose=0)[0]

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)


def run(input_path, output_path, train_autoencoder=False):
    np.random.seed(1234)

    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)

    gui_paths, img_paths = Dataset.load_paths_only(input_path)

    input_shape = dataset.input_shape
    output_size = dataset.output_size
    steps_per_epoch = dataset.size / BATCH_SIZE

    voc = Vocabulary()
    voc.retrieve(output_path)

    generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, input_shape=input_shape,
                                         generate_binary_sequences=True)

    # Included a generator for images only as an input for autoencoders
    generator_images = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE,
                                                input_shape=input_shape, generate_binary_sequences=True,
                                                images_only=True)


    if train_autoencoder:
        autoencoder_model = autoencoder_image(input_shape, input_shape, output_path)
        autoencoder_model.fit_generator(generator_images, steps_per_epoch=steps_per_epoch)
        clear_session()


    model = image_to_code(input_shape, output_size, output_path)
    model.fit_generator(generator, steps_per_epoch=steps_per_epoch)


@app.route("/Train")
def train():
    return render_template('Train.html')

@app.route("/TrainProcess", methods=["POST"])
def train_process():
    print("train_process Called")
    input_path = "static/datasets/web/all_data/training_features"
    output_path = "static/model"
    train_autoencoder = False


    run(input_path, output_path, train_autoencoder=train_autoencoder)
    return render_template('TrainProcess.html')


class Node:
    def __init__(self, key, value, data=None):
        self.key = key
        self.value = value
        self.data = data
        self.parent = None
        self.root = None
        self.children = []
        self.level = 0

    def add_children(self, children, beam_width):
        for child in children:
            child.level = self.level + 1
            child.value = child.value * self.value

        nodes = sorted(children, key=lambda node: node.value, reverse=True)
        nodes = nodes[:beam_width]

        for node in nodes:
            self.children.append(node)
            node.parent = self

        if self.parent is None:
            self.root = self
        else:
            self.root = self.parent.root
        child.root = self.root

    def remove_child(self, child):
        self.children.remove(child)

    def max_child(self):
        if len(self.children) == 0:
            return self

        max_childs = []
        for child in self.children:
            max_childs.append(child.max_child())

        nodes = sorted(max_childs, key=lambda child: child.value, reverse=True)
        return nodes[0]

    def show(self, depth=0):
        print(" " * depth, self.key, self.value, self.level)
        for child in self.children:
            child.show(depth + 2)


class BeamSearch:
    def __init__(self, beam_width=1):
        self.beam_width = beam_width

        self.root = None
        self.clear()

    def search(self):
        result = self.root.max_child()

        self.clear()
        return self.retrieve_path(result)

    def add_nodes(self, parent, children):
        parent.add_children(children, self.beam_width)

    def is_valid(self):
        leaves = self.get_leaves()
        level = leaves[0].level
        counter = 0
        for leaf in leaves:
            if leaf.level == level:
                counter += 1
            else:
                break

        if counter == len(leaves):
            return True

        return False

    def get_leaves(self):
        leaves = []
        self.search_leaves(self.root, leaves)
        return leaves

    def search_leaves(self, node, leaves):
        for child in node.children:
            if len(child.children) == 0:
                leaves.append(child)
            else:
                self.search_leaves(child, leaves)

    def prune_leaves(self):
        leaves = self.get_leaves()

        nodes = sorted(leaves, key=lambda leaf: leaf.value, reverse=True)
        nodes = nodes[self.beam_width:]

        for node in nodes:
            node.parent.remove_child(node)

        while not self.is_valid():
            leaves = self.get_leaves()
            max_level = 0
            for leaf in leaves:
                if leaf.level > max_level:
                    max_level = leaf.level

            for leaf in leaves:
                if leaf.level < max_level:
                    leaf.parent.remove_child(leaf)

    def clear(self):
        self.root = None
        self.root = Node("root", 1.0, None)

    def retrieve_path(self, end):
        path = [end.key]
        data = [end.data]
        while end.parent is not None:
            end = end.parent
            path.append(end.key)
            data.append(end.data)

        result_path = []
        result_data = []
        for i in range(len(path) - 2, -1, -1):
            result_path.append(path[i])
            result_data.append(data[i])
        return result_path, result_data


class Sampler:
    def __init__(self, voc_path, input_shape, output_size, context_length):
        self.voc = Vocabulary()
        self.voc.retrieve(voc_path)

        self.input_shape = input_shape
        self.output_size = output_size

        print("Vocabulary size: {}".format(self.voc.size))
        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

        self.context_length = context_length

    def predict_greedy(self, model, input_img, require_sparse_label=True, sequence_length=150, verbose=False):
        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Utils.sparsify(current_context, self.output_size)

        predictions = START_TOKEN
        out_probas = []

        for i in range(0, sequence_length):
            if verbose:
                print("predicting {}/{}...".format(i, sequence_length))

            probas = model.predict(input_img, np.array([current_context]))
            prediction = np.argmax(probas)
            out_probas.append(probas)

            new_context = []
            for j in range(1, self.context_length):
                new_context.append(current_context[j])

            if require_sparse_label:
                sparse_label = np.zeros(self.output_size)
                sparse_label[prediction] = 1
                new_context.append(sparse_label)
            else:
                new_context.append(prediction)

            current_context = new_context

            predictions += self.voc.token_lookup[prediction]

            if self.voc.token_lookup[prediction] == END_TOKEN:
                break

        return predictions, out_probas

    def recursive_beam_search(self, model, input_img, current_context, beam, current_node, sequence_length):
        probas = model.predict(input_img, np.array([current_context]))

        predictions = []
        for i in range(0, len(probas)):
            predictions.append((i, probas[i], probas))

        nodes = []
        for i in range(0, len(predictions)):
            prediction = predictions[i][0]
            score = predictions[i][1]
            output_probas = predictions[i][2]
            nodes.append(Node(prediction, score, output_probas))

        beam.add_nodes(current_node, nodes)

        if beam.is_valid():
            beam.prune_leaves()
            if sequence_length == 1 or self.voc.token_lookup[beam.root.max_child().key] == END_TOKEN:
                return

            for node in beam.get_leaves():
                prediction = node.key

                new_context = []
                for j in range(1, self.context_length):
                    new_context.append(current_context[j])
                sparse_label = np.zeros(self.output_size)
                sparse_label[prediction] = 1
                new_context.append(sparse_label)

                self.recursive_beam_search(model, input_img, new_context, beam, node, sequence_length - 1)

    def predict_beam_search(self, model, input_img, beam_width=3, require_sparse_label=True, sequence_length=150):
        predictions = START_TOKEN
        out_probas = []

        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Utils.sparsify(current_context, self.output_size)

        beam = BeamSearch(beam_width=beam_width)

        self.recursive_beam_search(model, input_img, current_context, beam, beam.root, sequence_length)

        predicted_sequence, probas_sequence = beam.search()

        for k in range(0, len(predicted_sequence)):
            prediction = predicted_sequence[k]
            probas = probas_sequence[k]
            out_probas.append(probas)

            predictions += self.voc.token_lookup[prediction]

        return predictions, out_probas



@app.route("/Produce")
def produce():
    return render_template('produce.html')

@app.route("/ProduceProcess", methods=["POST"])
def produce_process():
    print("produce_process Called")
    trained_weights_path = "static/model"
    trained_model_name = "imagetocode2"
    input_path = "static/datasets/web/eval"
    output_path = "static/guicode"
    search_method = "greedy"

    meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)
    input_shape = meta_dataset[0]
    output_size = meta_dataset[1]

    model = image_to_code(input_shape, output_size, trained_weights_path)
    model.load(trained_model_name)

    sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)

    for f in os.listdir(input_path):
        if f.find(".png") != -1:
            evaluation_img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)

            file_name = f[:f.find(".png")]

            if search_method == "greedy":
                result, _ = sampler.predict_greedy(model, np.array([evaluation_img]))
                print("Result greedy: {}".format(result))
            else:
                beam_width = int(search_method)
                print("Search with beam width: {}".format(beam_width))
                result, _ = sampler.predict_beam_search(model, np.array([evaluation_img]), beam_width=beam_width)
                print("Result beam: {}".format(result))

            with open("{}/{}.gui".format(output_path, file_name), 'w') as out_f:
                out_f.write(result.replace(START_TOKEN, "").replace(END_TOKEN, ""))
    print("ProduceProcess Finished")
    return render_template('ProduceProcess.html')


@app.route("/Sample")
def sample():
    return render_template('Sample.html')

@app.route("/SampleProcess", methods=["POST"])
def sample_process():
    print("sample_process Called")
    trained_weights_path = "static/model"
    trained_model_name = "imagetocode2"
    input_path = "static/sample/test1.png"
    output_path = "static/sample"
    search_method = "greedy"
    meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)
    input_shape = meta_dataset[0]
    output_size = meta_dataset[1]

    model = image_to_code(input_shape, output_size, trained_weights_path)
    model.load(trained_model_name)

    sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)

    file_name = basename(input_path)[:basename(input_path).find(".")]
    evaluation_img = Utils.get_preprocessed_img(input_path, IMAGE_SIZE)

    if search_method == "greedy":
        result, _ = sampler.predict_greedy(model, np.array([evaluation_img]))
        print("Result greedy: {}".format(result))
    else:
        beam_width = int(search_method)
        print("Search with beam width: {}".format(beam_width))
        result, _ = sampler.predict_beam_search(model, np.array([evaluation_img]), beam_width=beam_width)
        print("Result beam: {}".format(result))

    with open("{}/{}.gui".format(output_path, file_name), 'w') as out_f:
        out_f.write(result.replace(START_TOKEN, "").replace(END_TOKEN, ""))
    print("Sample Process Finished")
    return render_template('SampleProcess.html')



class CNode:
    def __init__(self, key, parent_node, content_holder):
        self.key = key
        self.parent = parent_node
        self.children = []
        self.content_holder = content_holder

    def add_child(self, child):
        self.children.append(child)

    def show(self):
        print(self.key)
        for child in self.children:
            child.show()

    def render(self, mapping, rendering_function=None):
        content = ""
        for child in self.children:
            content += child.render(mapping, rendering_function)

        value = mapping[self.key]
        if rendering_function is not None:
            value = rendering_function(self.key, value)

        if len(self.children) != 0:
            value = value.replace(self.content_holder, content)

        return value

class Compiler:
    def __init__(self, dsl_mapping_file_path):
        with open(dsl_mapping_file_path) as data_file:
            self.dsl_mapping = json.load(data_file)

        self.opening_tag = self.dsl_mapping["opening-tag"]
        self.closing_tag = self.dsl_mapping["closing-tag"]
        self.content_holder = self.opening_tag + self.closing_tag

        self.root = CNode("body", None, self.content_holder)

    def compile(self, input_file_path, output_file_path, rendering_function=None):
        dsl_file = open(input_file_path)
        current_parent = self.root

        for token in dsl_file:
            token = token.replace(" ", "").replace("\n", "")

            if token.find(self.opening_tag) != -1:
                token = token.replace(self.opening_tag, "")

                element = CNode(token, current_parent, self.content_holder)
                current_parent.add_child(element)
                current_parent = element
            elif token.find(self.closing_tag) != -1:
                current_parent = current_parent.parent
            else:
                tokens = token.split(",")
                for t in tokens:
                    element = CNode(t, current_parent, self.content_holder)
                    current_parent.add_child(element)

        output_html = self.root.render(self.dsl_mapping, rendering_function=rendering_function)
        with open(output_file_path, 'w') as output_file:
            output_file.write(output_html)

class CUtils:
    @staticmethod
    def get_random_text(length_text=10, space_number=1, with_upper_case=True):
        results = []
        while len(results) < length_text:
            char = random.choice(string.ascii_letters[:26])
            results.append(char)
        if with_upper_case:
            results[0] = results[0].upper()

        current_spaces = []
        while len(current_spaces) < space_number:
            space_pos = random.randint(2, length_text - 3)
            if space_pos in current_spaces:
                break
            results[space_pos] = " "
            if with_upper_case:
                results[space_pos + 1] = results[space_pos - 1].upper()

            current_spaces.append(space_pos)

        return ''.join(results)

    @staticmethod
    def get_ios_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.digits + string.ascii_letters)
            results.append(char)

        results[3] = "-"
        results[6] = "-"

        return ''.join(results)

    @staticmethod
    def get_android_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.ascii_letters)
            results.append(char)

        return ''.join(results)

@app.route("/CreateHTML")
def createHTML():
    return render_template('CreateHTML.html')

@app.route("/CreateHTMLProcess", methods=["POST"])
def create_HTML_process():
    print("createHTML_process Called")

    input_file = "static/sample/test1.gui"
    FILL_WITH_RANDOM_TEXT = True
    TEXT_PLACE_HOLDER = "[]"

    dsl_path = "static/web-dsl-mapping.json"
    compiler = Compiler(dsl_path)

    def render_content_with_text(key, value):
        if FILL_WITH_RANDOM_TEXT:
            if key.find("btn") != -1:
                value = value.replace(TEXT_PLACE_HOLDER, CUtils.get_random_text())
            elif key.find("title") != -1:
                value = value.replace(TEXT_PLACE_HOLDER, CUtils.get_random_text(length_text=5, space_number=0))
            elif key.find("text") != -1:
                value = value.replace(TEXT_PLACE_HOLDER,
                                      CUtils.get_random_text(length_text=56, space_number=7, with_upper_case=False))
        return value

    file_uid = basename(input_file)[:basename(input_file).find(".")]
    path = input_file[:input_file.find(file_uid)]
    print(file_uid, path)

    input_file_path = "{}{}.gui".format(path, file_uid)
    output_file_path = "{}{}.html".format(path, file_uid)
    compiler.compile(input_file_path, output_file_path, rendering_function=render_content_with_text)

    print("Create HTML Finished")
    return render_template('CreateHTMLProcess.html')


if __name__ == "__main__":
    app.run()

