from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_HArray1OfHAsciiString
from OCC.Core.APIHeaderSection import APIHeaderSection_MakeHeader
from OCC.Core.TCollection import TCollection_HAsciiString
from bspy import Solid, Boundary, Manifold
import BSpyConvert.convert as convert

def export_step(fileName, object):
    # Initialize the STEP writer.
    step_writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP203")

    if isinstance(object, (Solid, Boundary, Manifold)):
        objects = [object]
    else:
        objects = object

    objectCount = 1
    for object in objects:
        if isinstance(object, Manifold):
            name = f"Face {objectCount}"
            if hasattr(object, "metadata"):
                name = object.metadata.get("Name", name)
            Interface_Static.SetCVal("write.step.product.name", name)
            surface, flipNormal, transform = convert.convert_manifold_to_surface(object)
            face = convert.convert_surface_to_face(surface, flipNormal)
            step_writer.Transfer(face, STEPControl_AsIs)
        elif isinstance(object, Boundary):
            name = f"Face {objectCount}"
            if hasattr(object.manifold, "metadata"):
                name = object.manifold.metadata.get("Name", name)
            Interface_Static.SetCVal("write.step.product.name", name)
            face = convert.convert_boundary_to_face(object)
            step_writer.Transfer(face, STEPControl_AsIs)
        elif isinstance(object, Solid):
            shape = convert.convert_solid_to_shape(object)
            step_writer.Transfer(shape, STEPControl_AsIs)
        objectCount += 1

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
            solids.append(convert.convert_shape_to_solid(shape))
    
    return solids
