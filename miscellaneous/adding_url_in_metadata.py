import os
import json
import re
import re
import unicodedata
import time
data=pd.read_csv("UN_pdfs_links.csv")

for subdir, dirr, files in os.walk("take 2/Law_Index_01_12/metadatas"):
    for file in files:
        if file.endswith('.txt'):
            index = 0
            casename = str(file.split(" _")[0]).encode()
            for i in data["case_name"]:
                i=i.encode()
                casename_1 = unicodedata.normalize('NFC', casename.decode('utf-8'))
                casename_2 = unicodedata.normalize('NFC', i.decode('utf-8'))
                if casename_1 == casename_2:
                    case_url = data.iloc[index].case_url
                    pdf_url = data.iloc[index].pdf_url
                    try:
                        file_path = os.path.join(subdir, str(file))
                        with open(file_path) as f:
                            j_data = json.load(f)
                        j_data.update({"case_url": case_url})
                        j_data.update({"pdf_url": pdf_url})
                        
                        output_filename = os.path.join(subdir,"final" ,casename_1 + ".json")
                        print(output_filename)
                        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                        
                        with open(output_filename, "w") as output_file:
                            json.dump(j_data, output_file)
                    except Exception as e:
                        print(casename)
                        print("Error ", e)
                index += 1