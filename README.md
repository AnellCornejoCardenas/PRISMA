# PRISMA
Producing Resolved and Integrated Spectra from siMulated gAlaxies


This repository includes a Jupyter notebook that shows how to generate synthetic spectra using PRISMA. The methodology behind PRISMA is described in detail in the paper: **Mocking IFU observations and metallicity diagnostics from cosmological simulations** 
(Cornejo-Cárdenas et al. 2025, https://arxiv.org/abs/2506.00460).

**To run this tutorial, follow these steps:**

   1. **Create a Conda environment to work with PRISMA**
	You can install Anaconda from https://www.anaconda.com/download/success. 
	To create a Conda environment with Python 3.9.16, run the following command 
	(this example uses prisma_py3.9 as the environment name)

	conda create -n pprisma_py3.9 python=3.9.16


2- **Activate the environment:**

	conda activate prisma_py3.9


3- **Install CIGALE** by following the step-by-step installation 
   guide available at: https://cigale.lam.fr/documentation/. For 
   this tutorial, we used the version 2025.0

4- **Install additional dependencies with pip:**

	pip install jupyter pandas pytest-shutil shapely 

   
5- **Time to try PRISMA!**
	Navigate to the PRISMA directory (cd path/to/PRISMA) and run the notebook PRISMA.ipynb 
 	using the data provided in the Data_sims/ folder. Additionally, the notebook 
  	PRISMA_Results_Visualization.ipynb in the Utils/ directory shows how to display 
   	PRISMA results.
