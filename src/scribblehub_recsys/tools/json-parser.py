"""Small python script to parse json strings and output them as a CSV. Deprecated.
"""
import pandas as pd
import json

filename = input("Type the filename which you want to parse:")
column = input("Type the column which contains json string:")
#filename = "scribblehub-novels.xlsx"
#column = "genres"

def JSONParser(data):
    return json.loads(data)

df = pd.read_excel(filename,sheet_name="Sheet1")

df[column] = df[column].apply(JSONParser)

d = {}
i = 0

for index, row in df.iterrows():
    novel_title = row["title"]
    novel_link = row["novel-link-href"]
    for entry in row[column]:
        d[i] = {"title":novel_title, "novel-link-href":novel_link, column:entry[column]}
        i += 1
        

result = pd.DataFrame.from_dict(d, orient="index")
result.to_csv(column + "_from_json.csv", sep='~', index=False)
    