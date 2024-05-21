# MIMO Lab tools package

This provides a collection of functions and routines for standard data processing and analyses used in the lab.

### Setting up mimo_pack

Create a conda environment for working with mimo_pack.

In the terminal navigate to the mimo_pack directory and then enter:
> conda env create -f environment.yml`

This should create a conda environment named mimo_env, which can be used for working with mimo_pack tutorials. To start working the the code immediately, in the same terminal run:
> conda activate mimo_env

> jupyter notebook

### Adding your own code

mimo_pack is intended to provide an environment and suite of functions that facilitate analysis of your own data. To add your own code, create a folder in the mimo_pack directory called 'mycode' and add your scripts to it. From there, you can access the functions in mimo_pack using relative file imports. If you want to do something very similar to what is carried out in the tutorials, you can copy a tutorial into your 'mycode' directory and then modify it to fit your needs.