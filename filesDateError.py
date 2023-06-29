import os
import glob
import pandas as pd
import re
from datetime import datetime

directories = glob.glob('6월~12월_학생회관/*')

for directory in directories:
    files = sorted(os.listdir(directory))
    
    for filename in files:
        if filename.endswith(".xlsx"):
            absolute_filepath = os.path.join(directory, filename)

            data = pd.read_excel(absolute_filepath)

            date_str = data.iat[1, 0]

            match = re.search(r'(\d{4})년 (\d{2})월 (\d{2})일', date_str)

            if match is None:
                print(f"No date pattern found in file: {absolute_filepath}")
                continue

            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))

            date = datetime(year, month, day)

            formatted_date = datetime.strftime(date, "%d").lstrip("0")
            new_filename = formatted_date + '.xlsx'

            new_directory = os.path.join('6월~12월_학생회관', str(month) + '월')
            if not os.path.exists(new_directory):
                os.makedirs(new_directory)

            new_filepath = os.path.join(new_directory, new_filename)
            
            os.rename(absolute_filepath, new_filepath)
