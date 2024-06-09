from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_HArray1OfHAsciiString
from OCC.Core.APIHeaderSection import APIHeaderSection_MakeHeader
from OCC.Core.TCollection import TCollection_HAsciiString
from bspy import Solid, Spline
from BSpyConvert.convert import convert_spline_to_face, convert_shape_to_solid

def export_step(fileName, solid):
    if isinstance(solid, Solid):
        if solid.dimension != 3: raise ValueError("Solid must be 3D (dimension == 3)")
        if solid.containsInfinity: raise ValueError("Solid must be finite (containsInfinity == False)")
        splines = [boundary.manifold for boundary in solid.boundaries]
    elif isinstance(solid, Spline):
        splines = [solid]
    else:
        splines = solid

    # Initialize the STEP writer.
    step_writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP203")

    # Transfer spline faces.
    faceCount = 1
    for spline in splines:
        name = spline.metadata.get("Name", f"Face {faceCount}")
        Interface_Static.SetCVal("write.step.product.name", name)
        step_writer.Transfer(convert_spline_to_face(spline), STEPControl_AsIs)
        faceCount += 1

    # Create STEP header.
    model = step_writer.Model()
    model.ClearHeader()

    header = APIHeaderSection_MakeHeader()
    header.SetName(TCollection_HAsciiString("BSpy Spline"))
    header.SetAuthorValue(1, TCollection_HAsciiString("Eric Brechner"))
    header.SetAuthorisation(TCollection_HAsciiString("BSpy (c) 2024"))

    description = Interface_HArray1OfHAsciiString(1, 1)
    description.SetValue(1, TCollection_HAsciiString("A b-spline surface produced by BSpy"))
    header.SetDescription(description)

    org = Interface_HArray1OfHAsciiString(1, 1)
    org.SetValue(1, TCollection_HAsciiString("BSpy organization"))
    header.SetOrganization(org)

    header.SetOriginatingSystem(TCollection_HAsciiString("BSpyConvert"))
    #header.SetImplementationLevel(TCollection_HAsciiString("implementation level"))

    identifiers = Interface_HArray1OfHAsciiString(1, 1)
    identifiers.SetValue(1, TCollection_HAsciiString("OpenCascade (pythonocc)"))
    header.SetSchemaIdentifiers(identifiers)

    #header.SetPreprocessorVersion(TCollection_HAsciiString("preprocessor version"))
    #header.SetTimeStamp(TCollection_HAsciiString(f"Time stamp: {datetime.now()}"))

    model.AddHeaderEntity(header.FnValue())
    model.AddHeaderEntity(header.FsValue())
    model.AddHeaderEntity(header.FdValue())

    status = step_writer.Write(fileName)

    if status != IFSelect_RetDone:
        raise AssertionError("Write failed")

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
            solids.append(convert_shape_to_solid(shape))
    
    return solids
