import sqlite3
import pandas as pd
import json

# creating file path
DB_FILE = 'D:\projects\openclassrooms\projets\P7_geran_laurent\homecredit_data\db.db'


# creating cursor
con = sqlite3.connect(DB_FILE)
cur = con.cursor()

def localLoad(table):
    cur.execute("SELECT * FROM "+table)
    columns = cur.description 
    result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cur.fetchall()]
    resultJSON = json.dumps(result)
    df = pd.read_json(resultJSON,orient ='records')
    return (df)

applicationTrain = localLoad('application_train')
applicationTest = localLoad('application_test')

print(applicationTest.head())

#con.close()

#columns = cur.description 
#result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cur.fetchall()]
# Be sure to close the connection
