from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
import BSpyConvert.convert as convert

def import_step(fileName):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(fileName)
    if status != IFSelect_RetDone:
        raise ValueError("Can't read file.")
    if not step_reader.TransferRoots():
        raise ValueError("Transfer failed.")

    solids = []
    for i in range(step_reader.NbShapes()):
        shape = step_reader.Shape(i + 1)
        if not shape.IsNull():
            solids.append(convert.convert_shape_to_solid(shape))
    
    return solids
