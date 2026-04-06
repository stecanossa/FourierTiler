# **Welcome to Fourier Tiler!**


This program provides an image-based analogy of how diffraction and total scattering works, which considers:

•   Unit cell content as images (tiles), therefore pixel brightness as scatterer density.  
•   Substitutional disorder as alternative tiles, representing unit cells with different contents.  
•   Scattering patterns as squared Fourier Transform (FT) of the crystal's structure (tiling).  


In the most basic workflow, the program executes these sequential processes:
1. Loads images that are used as tiles to assemble a square tiling.
2. Initiates a Monte Carlo simulation to produce a disordered tiling (or ordered, if only one tile is loaded).
3. Produces a squared FT of the tiling image — |FT(image)|² — representing its "diffraction pattern".

Additional functions include the inspection of the squared difference form factor of a pair of tiles, the average structure of the tiling, and the definition of correlation parameters to control local ordering of tiles in non-periodic mosaics, as well as several customization options for the output images.                 
<br><br>

# Use instructions

Fourier Tiler is provided either as (1) Windows executable or (2) python script, which do not require any installation.    
The first option, available within a forthcoming release, gives access to a graphical user interface facilitating the use of the program, although without the freedom of modifying the underlying code.
The second option, still user friendly and easy to use, is kept light by not having any user interface and allows anyone to modify and customize its code.


### Running the Windows version

Double click on the ".exe" file. In case guidance is needed, users can follow the instructions provided in the User Manual (herewith available for download).


### Running the Python version
N.B. The following python modules are required for running the script: **sys, os, random, datetime, math, numpy, PIL (Pillow)**

1- Make sure you have "FourierTiler.py" and "Input.txt" in the same folder.    
2- Update "Input.txt" file with the path(s) and name(s) of the input image(s) for the tiling, at the line starting with "tile = ".    
3- Modify the default options if you like, and add optional interactions at the end of the file.    
4- Open a command line and navigate to the folder where you have the program and input file.    
5- Execute "python fouriertiler.py Input.txt" (the input.txt file is automatically loaded if not specified in the command).  
<br><br>

## Tile images

A "My Tilebox" folder with some example tile images is provided among the available files, as zipped file. Tiles can have any size, shape or content: the program automatically fixes size or shape mismatches automatically. Accepted formats are JPG, JPEG, PNG, BMP, and TIFF.Users are highly encouraged to crate their custom tiles and let their creativity run free!   
Further useful information can be found in the User Manual provided within the release.
<br><br>

#   Acknowledgements and Permissions
   
Fourier Tiler has been developed thanks to the kind support by the EU4MOFs network (COST action CA22147) via an awarded Virtual Mobility (VM) grant.
Coding assistance by Microsoft Copilot and Anthropic Claude were essential for its development, while design of interface, functions, and behaviour have been independent from AI contribution.   
The use, distribution, and modification of the files here disclosed for non-commercial purposes are allowed upon acknowledging the program's author (Stefano Canossa), any additional contributors, and supporting parties. For more information, please read the "licence information" file. 
