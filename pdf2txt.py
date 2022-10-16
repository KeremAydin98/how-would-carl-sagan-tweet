import PyPDF2

# create file object variable
# opening method will be rb
pdffileobj = open('.Data/cosmos.pdf', 'rb')

# create reader variable that will read the pdffileobj
pdfreader = PyPDF2.PdfFileReader(pdffileobj)

# This will store the number of pages of this pdf file
x = pdfreader.numPages

all_text = ""
for i in range(4,x-4):

    # create a variable that will select the selected number of pages
    pageobj = pdfreader.getPage(i)

    # create text variable which will store all text datafrom pdf file
    text = pageobj.extractText()

    all_text = all_text + "\n" + text

# Write all the text in one txt file
with open("./Data/all_text.txt", "w") as f:
    f.writelines(all_text)