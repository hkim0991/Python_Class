# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 09:19:20 2018

@author: 202-22
"""

# Import libraries & funtions ------------------------------------------------
import os 
import pandas as pd
import pymongo
import datetime, time


# Connect to mongoDB ----------------------------------------------------------
mng_client = pymongo.MongoClient('localhost', 27017)  # connect to mongoDB

mng_client.database_names()
db = mng_client.testDB06  # create/use the database
db_col = db.col01  # create collection 
posts = db.posts 

date=datetime.date.today()
date


# Data insertion in a collection ----------------------------------------------
db_col.insert( {'author':'Tomi', 
                                   'test':'My first blog post!', 
                                   'tags':['mongodb', 'python', 'data'], 
                                   'date': datetime.datetime.utcnow()} )

posts.insert_many([{'author':'Mike',
                                   'test':'Another post!',
                                   'tags':['bulk', 'insert'],
                                   'date': datetime.datetime(2018, 6, 8, 10, 11)},
                                    {'authoer':'Eliot',
                                     'title': 'Python is fun',
                                     'test':'Today is good!',
                                     'date': datetime.datetime(2018, 6, 8, 10, 11)}])


# Data check ------------------------------------------------------------------
curso1 = posts.find()
for i in curso1:
    print(i)
    
print(posts.find().count())  #result = 3

print(posts.find({"author":"Mike"}).count())  #3


# create a function -----------------------------------------------------------
def get_db():
    connect = pymongo.MongoClient('localhost', 27017)
    db1 = connect.testDB06
    return db1

def add_country(db1):
    db1.countries.insert({'name':'canada'})
    
def get_country(db1):
    return db1.countries.find_one()

if __name__ == "__main__":
    db = get_db()
    add_country(db)
    print(get_country(db))

    
    
