import os, string
import PyPDF2

all_text = ""
# create file object variable
# opening method will be rb
for file in os.listdir("Data/"):
    if file.endswith(".pdf"):
        pdffileobj = open(f'Data/{file}', 'rb')

        # create reader variable that will read the pdffileobj
        pdfreader = PyPDF2.PdfFileReader(pdffileobj)

        # This will store the number of pages of this pdf file
        x = pdfreader.numPages

        for i in range(4,x-4):

            # create a variable that will select the selected number of pages
            pageobj = pdfreader.getPage(i)

            # create text variable which will store all text datafrom pdf file
            text = pageobj.extractText()

            text = text.encode("ascii","ignore")
            text = text.decode()

            punctuations = string.punctuation
            punctuations.replace(",","")
            punctuations.replace(".","")

            text = text.translate(str.maketrans('', '', punctuations))

            all_text = all_text + "\n" + text

# Write all the text in one txt file
with open("./Data/all_text.txt", "w") as f:
    f.writelines(all_text)