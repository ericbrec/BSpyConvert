
from bspy import Solid
from BSpyConvert.step import export_step

if __name__ == "__main__":
    solids = Solid.load(r"c:\users\ericb\onedrive\desktop\Solid.json")
    export_step(r"c:\users\ericb\onedrive\desktop\Solid.stp", solids)