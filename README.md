# Measuring the Higgs boson self-coupling at the FCC-hh using AI
Project for the 2026 Physics Work Experience Week at the University of Liverpool on using machine learning to classify simulated FCC di-Higgs production data.
This project aims at looking at simulated Higgs pairs which subsequently decay to a b-quark pair and tau lepton pair. This is compared against against a background sample of a process known as ttbar, where a pair of top-quarks decay into a b-quark each, via W bosons which decay into tau leptons and neutrinos.

These samples will first be looked at through taking simple cuts on some variables that are calculated from the final-state object coordinates, or kinematics, within the detector. These cuts will be used to calculate a significance value that we use to measure how statistically distinct the signal process is from background.

Then we will move onto working with simple neural networks which take the input data and classifies the data into being more signal or background-like. We will work on making improvements to this neural network and try and get a better value of significance comapred to the previous method.

The project involves some basic programming in Python, however, no experience with Python or any coding is needed for this project. Hopefully, this project will allow you to explore some basic concepts in collider-based particle physics, as well as some basic concepts of machine learning.

## First time setup:

If you dont have conda installed, download it from here: https://www.anaconda.com/docs/getting-started/installation 

### Windows

1. On Windows, open **PowerShell** or **Anaconda PowerShell Prompt** after you have logged into your account.

2. Choose a folder where you want to keep the project, move into it, and clone the repository. For example:
```
mkdir PhysicsWorkExperienceWeek
cd PhysicsWorkExperienceWeek
git clone https://github.com/jdegens/fcc_ml_project.git
```
  You can choose a different folder name or location if you prefer. Please ask me in case you are not sure.

3. Check that conda is available. If `conda --version` fails, install Miniconda first.



4. Move into the project directory and create the python environment:
```
cd fcc_ml_work_experience_liverpool
conda env create -f environment.yml
conda activate FCCHH
pip install -e .
jupyter lab
```
  This installs all the required packages and will take 10-15 minutes. You will see package names scrolling past — this is normal.

5. Activate the environment:
```
conda activate FCCHH
```
  You should see `(FCCHH)` appear at the start of the prompt line.

6. Launch Jupyter Lab:
```
jupyter lab
```
  JupyterLab will open automatically in your browser.

### macOS/Linux

1. Open a terminal.

2. Choose a folder where you want to keep the project, move into it, and clone the repository. For example:
```
mkdir PhysicsWorkExperienceWeek
cd PhysicsWorkExperienceWeek
git clone https://github.com/LennoxWood/fcc_ml_work_experience_liverpool.git
```

3. Check that conda is available. If `conda --version` fails, install Miniconda first:

4. Move into the project directory and create the Python environment:
```
cd fcc_ml_work_experience_liverpool
conda env create -f environment.yml
conda activate FCCHH
pip install -e .
jupyter lab
```
  This installs all the required packages and will take 10-15 minutes. You will see package names scrolling past — this is normal.

## Subsequent setup:

### Windows

1. Open **PowerShell** or **Anaconda PowerShell Prompt**.
2. Move into the cloned project folder, then run:
```
cd path\to\fcc_ml_work_experience_liverpool
conda activate FCCHH
jupyter lab
```
  Replace `path\to\fcc_ml_work_experience_liverpool` with the folder where you cloned the repository.

### macOS/Linux

1. Open a terminal.
2. Move into the cloned project folder, then run:
```
cd /path/to/fcc_ml_work_experience_liverpool
conda activate FCCHH
jupyter lab
```
  Replace `/path/to/fcc_ml_work_experience_liverpool` with the folder where you cloned the repository.

Hope you also find this project fun :)


