from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
import BSpyConvert.convert as convert

def import_iges(fileName):
    reader = IGESControl_Reader()
    status = reader.ReadFile(fileName)
    if status != IFSelect_RetDone:
        raise ValueError("Can't read file.")
    if not reader.TransferRoots():
        raise ValueError("Transfer failed.")

    solids = []
    for i in range(reader.NbShapes()):
        shape = reader.Shape(i + 1)
        if not shape.IsNull():
            solids.append(convert.convert_shape_to_solid(shape))
    
    return solids

def import_step(fileName):
    reader = STEPControl_Reader()
    status = reader.ReadFile(fileName)
    if status != IFSelect_RetDone:
        raise ValueError("Can't read file.")
    if not reader.TransferRoots():
        raise ValueError("Transfer failed.")

    solids = []
    for i in range(reader.NbShapes()):
        shape = reader.Shape(i + 1)
        if not shape.IsNull():
            solids.append(convert.convert_shape_to_solid(shape))
    
    return solids
