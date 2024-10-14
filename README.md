# IDC_SQL_lib
Installation Guide:
For the purposes of this guide, VS Code will be used as IDE of choice.
1. Install Jupyter
1.1 Use Anaconda to install Jupyter 
2. Install Python in VS Code and to your local PC
3. Press ctrl+shift+p to run a VS Code command 
4. Search Python: Select Interpreter ![alt text](https://i.gyazo.com/638b9c35015c0ca23b790fe537704bd0.png)
5. Press "Create Virtual Environment"
5.1 Then choose Venv
5.2 Choose an installation of Python
6. Make sure that requirements.txt is ticked off ![alt text](https://i.gyazo.com/12aad1fa8c1ed66ee66d258aacd1d432.png)
7. Press ok
8. After completion, try running Main.py
9. Open a terminal and input .\.venv\Scripts\activate
10. Run the following input python -m ipykernel install --user --name=PUTENVNAMEHERE --display-name "PUTENVNAMEHERE"
11. Now open any Jupyter Notebook through Anaconda and select the newly created kernel