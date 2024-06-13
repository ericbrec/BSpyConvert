from bspy import Viewer
from BSpyConvert.step import import_step

if __name__ == "__main__":
    viewer = Viewer()

    solids = import_step(r"c:\users\ericb\onedrive\desktop\Solid.stp")
    for solid in solids:
        viewer.draw(solid)
    viewer.mainloop()
