Welcome to FourierTiler!

This program provides an image-based analogy of how diffraction and total scattering works, which considers:

•   Unit cell content as images (tiles), therefore pixel brightness as scatterer density.
•   Substitutional disorder as alternative tiles, representing unit cells with different contents.
•   Scattering patterns as squared Fourier Transform (FT) of the crystal's structure (tiling).

In the most basic workflow, the program executes these sequential processes:
1. Loads images that are used as tiles to assemble a square tiling.
2. Initiates a Monte Carlo simulation to produce a disordered tiling (or ordered, if only one tile is loaded).
3. Produces a squared FT of the tiling image — |FT(image)|² — representing its "diffraction pattern".

Additional functions include the inspection of the squared difference form factor of a pair of tiles, the average structure of the tiling, and the definition of correlation parameters to control local ordering of tiles in non-periodic mosaics, as well as several customization options for the output images.

---------------
  HOW TO USE
---------------

Fourier Tiler is provided either as (1) Windows executable or (2) python script. 
The first option gives access to a Graphical User Interface facilitating the use of the program, while preventing more coding-savvy users from modifying it.
The second option is still very user friendly and easy to use for any user, but is kept light by not having any user interface (except for the simple command line). This version allows anyone to modify the underlying code within the provided licence regulations.

Running the Windows version: Double click on the ".exe" file. In case guidance is needed, users can follow the instructions provided in the User Manual (herewith available for download).

Running the Python version: 
(N.B. The python modules sys, os, random, io, datetime, numpy, PIL, matplotlib, and scipy should be installed and working in the environment used for running Fourier Tiler.)

1- Make sure you have "FourierTiler.py" and "Input.txt" in the same folder.
2- Update "Input.txt" file with the path(s) and name(s) of the input image(s) for the tiling, at the line starting with "tile = ".
3- Modify the default options if you like, and add optional interactions at the end of the file.
4- Open a command line and navigate to the folder where you have the program and input file.
5- Execute "python fouriertiler.py" (the input.txt is automatically loaded if not specified after the script).

-------------------------------------------------
   ACKNOWLEDGEMENTS and COPYRIGHT INFORMATION
-------------------------------------------------

Fourier Tiler has been developed by Stefano Canossa thanks also to the kind support by EU4MOFs COST action (CA22147) via an awarded Virtual Mobility (VM) grant.
The use, distribution, and modification of the provided files and codes are allowed within the boundaries of a CC BY-NC-SA licence, which prohibits unauthorized commercial uses of the program and its derivatives or outputs, and require acknowledgements to its author and contributing parties when make public use of the program or its output files.

For more information, please read the "licence information" file. 
