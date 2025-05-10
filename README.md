# federated-learning-project

# Before running

if not already installed,

- download python version 3.10 (https://www.python.org/downloads/)
  remember to install certificates after install, otherwise you get an error when running the code.
  (SSL certificates problem on macos --> open a terminal and run: /Applications/Python\ 3.10/Install\ Certificates.command)

- python3.10 -m venv venv
- (macos) source venv/bin/activate | (windows) venv/Scripts/activate
- python3.10 -m pip install --upgrade pip
- pip install -r requirements.txt

# Running scripts

- cd src
- python main.py

# Running notebooks

Go to https://colab.research.google.com/
Select: Github >
Then:

- Past the repository URL or select a user or organization
- Select the Repository name
- Select the Branch of the notebooks you want to run
- Open the notebook of you interest and run it. (Example: click on _src/notebooks/test.ipynb_)

# Resources

### google colab integration with Github

https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb#scrollTo=8J3NBxtZpPcK
