from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
import BSpyConvert.convert as convert

def import_iges(fileName):
    """
    Import an IGES file into BSpy.

    Parameters
    ----------
    fileName : `str`
        The path to the IGES file, including extension.

    Returns
    -------
    solids : `list` of `BSpy.Solid`
        The list of BSpy solids.
    """
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
            solids += convert.convert_shape_to_solids(shape)
    
    return solids

def import_step(fileName):
    """
    Import a STEP file into BSpy.

    Parameters
    ----------
    fileName : `str`
        The path to the STEP file, including extension.

    Returns
    -------
    solids : `list` of `BSpy.Solid`
        The list of BSpy solids.
    """
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
            solids += convert.convert_shape_to_solids(shape)
    
    return solids
