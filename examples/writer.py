
from bspy import Spline
from BSpyConvert.step import export_step

if __name__ == "__main__":
    splines = Spline.load(r"c:\users\ericb\onedrive\desktop\Surface1.json")
    export_step(r"c:\users\ericb\onedrive\desktop\Surface1.stp", splines)