from bspy import Viewer
from BSpyConvert.importer import import_step, import_iges

if __name__ == "__main__":
    viewer = Viewer()

    solids = import_step(r"C:\Users\ericb\OneDrive\Documents\Publications\B-rep booleans\Island.stp")
    #solids = import_step(r"c:\users\ericb\onedrive\desktop\Solid.stp")
    #solids = import_iges(r"c:\users\ericb\onedrive\desktop\Solid.igs")
    for solid in solids:
        viewer.draw(solid)
    viewer.mainloop()
