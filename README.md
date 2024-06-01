# MIMO Lab tools package

This provides a collection of functions and routines for standard data processing and analyses used in the lab. It is intended as a starting point for your own project.

### Setting up mimo_pack

Download the zip file of mimo_pack using the green button above.

Unzip the package and place the contents of the folder 'mimo_pack-main' inside a folder with the name of your project.

In the command, navigate to that folder with your project name. Then, create a conda environment for working with mimo_pack.

> conda env create -n PROJECT_NAME -f environment.yml`

where `PROJECT_NAME` is the name you want to give to the environment. You should create a new environment for each project. 

Upon completion, a conda environment named `PROJECT_NAME` is created that can be used as starting point for your own project. Additional packages can be added to it as needed [(e.g. see here)](https://stackoverflow.com/questions/33680946/how-to-add-package-to-conda-environment-without-pip).

Once the environment is created, activate it with the command:
> conda activate mimo_env

and then run the following command to make the code base accesible:
> pip install -e .

### Exploring the code base
To start working the the code base immediately, in the same terminal run:
> jupyter notebook

From there, you can load the tutorials to familiarize yourself with types of analyses you can do.

### Adding your own code

mimo_pack is intended to provide an environment and suite of functions that facilitate analysis of your own data.  To add your own code, create a folder in the mimo_pack directory called 'project' and add your scripts to it. From there, you can access the functions in mimo_pack using import statements. If you want to do something very similar to what is carried out in the tutorials, you can copy a tutorial into your 'project' directory and then modify it to fit your needs.