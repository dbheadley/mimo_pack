# Running spike sorting
This document covers how to run spike sorting.

## Installation
Several pieces of software need to be installed before you can begin.
1. Create a conda environment for Kilosort4 by following the instructions on its [github page](https://github.com/MouseLand/Kilosort).
2. Go the the [SpikeGLX site](https://billkarsh.github.io/SpikeGLX/) and download CatGT. For a detailed help of the processing options it offers, [see here](https://billkarsh.github.io/SpikeGLX/help/syncEdges/Sync_edges/#catgt-event-extraction). To better understand how it works, you can read the explanation [here](https://billkarsh.github.io/SpikeGLX/help/catgt_tshift/catgt_tshift/)
3. Once CatGT is downloaded, make it callable from the command line. The instructions for doing this on Windows are:
    1. Extract the CatGT-win folder and move it to the *C:\Program Files* directory.
    2. Type 'environment variables' into the search box in the Windows task bar and click the option 'Edit the system environment variables'.
    3. Click the 'Environment Variables...' button at the bottom of the window that pops up.
    4. Under the panel heading 'System variables', click on the line whose Variable name is 'Path' (the line should turn gray), and then press the 'Edit...' button.
    5. In the window that pops up, click the 'New' button. Enter the directory where CatGT was placed (*C:\Program Files\CatGT-win*).
    6. Verify CatGT works by opening a terminal (type 'Command Prompt' into the Windows search bar) and entering 'CatGT'. If the environment variable was successfully modified after pressing enter the terminal should return a new command line and in the directory you called CatGT a log file will be created named *CatGT.log* that lists the configuration settings for CatGT.
4. Download *SGLXMetaToCoords.py* from its [github page](https://github.com/jenniferColonell/SGLXMetaToCoords/blob/main/SGLXMetaToCoords.py). Place it on the drive where you intend to place the files for sorting.
5. Download *SpikeGLX_Datafile_Tools* from its [github page](https://github.com/jenniferColonell/SpikeGLX_Datafile_Tools). Place the folder *DemoReadSGLXData* found in the *Python* directory on the drive where you intend to place the files for sorting. For a demonstration of how to use the functions contained in *readSGLX.py* check out this Jupyter notebook for [analog signals](https://github.com/jenniferColonell/SpikeGLX_Datafile_Tools/blob/main/Python/read_SGLX_analog.ipynb) or this one for [digital signals](https://github.com/jenniferColonell/SpikeGLX_Datafile_Tools/blob/main/Python/read_SGLX_digital.ipynb).

