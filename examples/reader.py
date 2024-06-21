import logging
import numpy as np
from bspy import Viewer
from BSpyConvert.importer import import_step, import_iges

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s', datefmt='%H:%M:%S')
    np.set_printoptions(suppress=True)

    logging.info("Import file")
    solids = import_step(r"C:\Users\ericb\Downloads\radiator impeller.STEP")
    #solids = import_step(r"C:\Users\ericb\OneDrive\Documents\Publications\B-rep booleans\Island.stp")
    #solids = import_step(r"c:\users\ericb\onedrive\desktop\Solid.stp")
    #solids = import_iges(r"c:\users\ericb\onedrive\desktop\Solid.igs")

    logging.info("Render solids")
    viewer = Viewer()
    for solid in solids:
        viewer.draw(solid)
    viewer.mainloop()
