# federated-learning-project

# Clone the repo

git clone https://github.com/giovanna-brod-zamojska/federated-learning-project.git

# Before running

if not already installed,

- download python version 3.11 (https://www.python.org/downloads)
- python3.11 -m venv venv
- (macos) source venv/bin/activate | (windows) venv/Scripts/activate
- python3.11 -m pip install --upgrade pip
- pip install -r requirements.txt

# Running scripts

- cd src
- python main.py

# Running notebooks

Go to https://colab.research.google.com/
Select: Github >
Then:

- Past the repository URL or select the user or the organization account that created the repo
- Select the Repository name
- Select the Branch of the notebooks you want to run
- Open the notebook of your interest and run it

When doing so, only the notebook is uploaded in that current runtime session.
Google colab isn't synchronizing with the whole repo, unfortunately.
Therefore, inside the notebook runtime, you need to reclone the entire repository.

An example on how to do is shown in _src/notebooks/test.ipynb_

At the moment our repository is private, so in the above test.ipynb you will be prompted to enter a Github Personal Access Token that has access to that repo.

To create a Personal Access Token:
Go on github: Setting > Developer Settings > Personal Access Tokens > Generate a new token
Give to it at least Read-only permissions on the "Contents" section under "Permissions"

# Resources

### Google Colab and Github

https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb#scrollTo=8J3NBxtZpPcK
