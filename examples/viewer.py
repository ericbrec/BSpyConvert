from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Display.SimpleGui import init_display

if __name__ == "__main__":
    display, start_display, add_menu, add_function_to_menu = init_display()

    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(r"c:\users\ericb\onedrive\desktop\Surface1.stp")
    if status != IFSelect_RetDone:
        raise ValueError("Can't read file.")
    if not step_reader.TransferRoots():
        raise ValueError("Transfer failed.")

    for i in range(step_reader.NbShapes()):
        shape = step_reader.Shape(i + 1)
        display.DisplayShape(shape)

    display.FitAll()
    start_display()