import os
import win32com.client as win32
import glob

directories = glob.glob('GWNU_C9_발전량/*')

excel = win32.gencache.EnsureDispatch('Excel.Application')

for directory in directories:
    files = sorted(os.listdir(directory))
    
    for index, filename in enumerate(files, start=1):
        if filename.endswith(".xls"):
            absolute_filepath = os.path.abspath(os.path.join(directory, filename))
            wb = excel.Workbooks.Open(absolute_filepath)
            
            new_filename = str(index) + '.xlsx'
            new_filepath = os.path.join(directory, new_filename)
            
            absolute_new_filepath = os.path.abspath(new_filepath)

            wb.SaveAs(absolute_new_filepath, FileFormat = 51)
            wb.Close()

excel.Application.Quit()
