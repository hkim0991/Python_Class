# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:27:10 2018

@author: kimi
"""

# Import libraries & funtions ------------------------------------------------
import os 
import pandas as pd
import pymongo
import json


# Connect to mongoDB ----------------------------------------------------------
mng_client = pymongo.MongoClient('localhost', 27017)  # connect to mongoDB
db = mng_client.testA  # choose  & use testA db
db_col = db.items  # create collection 


# create a function -----------------------------------------------------------
def import_content(filepath):
    file_res = filepath
    data = pd.read_csv(file_res)
    data_json = json.loads(data.to_json(orient='records'))  # data type of MongoDB
    db_col.remove()
    db_col.insert(data_json)

    
# example 1: kaggle yelp data -------------------------------------------------
filepath = 'E:/data/yelp_business_hours.csv'
import_content(filepath)

cursor = db_col.find().limit(10)
for doc in cursor:
    print(doc)
 

# example 2 : kaggle bike demanding data --------------------------------------
import_content('C:/Users/202-22/Documents/R/Bike_Demanding_Project/data/train.csv')
cursor = db_col.find().limit(10)
for doc in cursor:
    print(doc)

print(db_col.find({"count": {"$gt":5}}).count())  #10187


curso1 = db_col.find({"count": {"$gt":5}}).limit(15)
for i in curso1:
    print(i)

