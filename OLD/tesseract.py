# from PIL import Image
# import os
# import pytesseract
# import shutil
# directory = 'letter_dataset/'

# text = pytesseract.image_to_string(Image.open('letter_dataset/1684866947257846.jpg'), config='--psm 6')
# print('text:', str(text))

# for filename in os.listdir(directory):
#     path = directory + '/' + filename
#     text = pytesseract.image_to_string(Image.open(path), config='--psm 6')

#     if len(text) !=1:
#         continue
#     folder_name = text.upper()
#     os.makedirs(folder_name, exist_ok=True)

#     new_image_path = os.path.join(folder_name, os.path.basename(filename))
#     print('text: ', text)
#     print('old: ', path)
#     print('new: ', new_image_path)

#     # shutil.move(path, new_image_path)


import os
import shutil
import pytesseract

# Path to the Tesseract OCR executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Input and output folder paths
# input_folder = 'input'
# output_folder = 'output'

# Iterate over files and folders in the input folder



# for root, dirs, files in os.walk(input_folder):
#     # Iterate over files
#     for filename in files:
#         # Check if the file is an image
#         try:
#             if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 # Construct the file paths
#                 file_path = os.path.join(root, filename)
#                 output_subfolder = os.path.relpath(root, input_folder)
                
#                 # Create subfolders in the output folder
#                 output_subfolder_path = os.path.join(output_folder, output_subfolder)
#                 os.makedirs(output_subfolder_path, exist_ok=True)
                
#                 # Perform OCR on the image
#                 text = pytesseract.image_to_string(file_path, config='--psm 6')
#                 letter = text[0] if text else ''
#                 print('letter: ', letter)
#                 # Create a folder with the name of the letter (if it doesn't exist)
#                 letter_folder = os.path.join(output_subfolder_path, letter.upper())
#                 os.makedirs(letter_folder, exist_ok=True)
                
#                 # Move the file to the letter folder
#                 new_file_path = os.path.join(letter_folder, filename)
#                 shutil.copy(file_path, new_file_path)
#         except:
#             'error'
