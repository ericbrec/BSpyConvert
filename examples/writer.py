
from bspy import Solid
from BSpyConvert.exporter import export_step, export_iges, export_stl

if __name__ == "__main__":
    solids = Solid.load(r"c:\users\ericb\onedrive\desktop\Solid.json")
    export_step(r"c:\users\ericb\onedrive\desktop\Solid.stp", solids)
    export_iges(r"c:\users\ericb\onedrive\desktop\Solid.igs", solids)
    export_stl(r"c:\users\ericb\onedrive\desktop\Solid.stl", solids[0])    