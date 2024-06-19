from OCC.Core.IGESControl import IGESControl_Writer
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.Interface import Interface_HArray1OfHAsciiString
from OCC.Core.APIHeaderSection import APIHeaderSection_MakeHeader
from OCC.Core.TCollection import TCollection_HAsciiString
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.DataExchange import write_stl_file
from bspy import Solid, Boundary, Manifold
import BSpyConvert.convert as convert

def export_iges(fileName, object):
    writer = IGESControl_Writer()

    if isinstance(object, (TopoDS_Shape, Solid, Boundary, Manifold)):
        objects = [object]
    else:
        objects = object

    for object in objects:
        if isinstance(object, Manifold):
            surface, flipNormal, transform = convert.convert_manifold_to_surface(object)
            writer.AddGeom(surface)
        elif isinstance(object, Boundary):
            for face in convert.convert_boundary_to_faces(object):
                writer.AddShape(face)
        elif isinstance(object, Solid):
            shape = convert.convert_solid_to_shape(object)
            writer.AddShape(shape)
        elif isinstance(object, TopoDS_Shape):
            writer.AddShape(object)
        else:
            raise ValueError("Invalid object type")
    
    if not writer.Write(fileName):
        raise AssertionError("Write failed")

def export_step(fileName, object):
    writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP203")

    if isinstance(object, (TopoDS_Shape, Solid, Boundary, Manifold)):
        objects = [object]
    else:
        objects = object

    objectCount = 1
    for object in objects:
        if isinstance(object, Manifold):
            name = f"Manifold {objectCount}"
            if hasattr(object, "metadata"):
                name = object.metadata.get("Name", name)
            Interface_Static.SetCVal("write.step.product.name", name)
            surface, flipNormal, transform = convert.convert_manifold_to_surface(object)
            face = convert.convert_surface_to_face(surface, flipNormal)
            writer.Transfer(face, STEPControl_AsIs)
        elif isinstance(object, Boundary):
            name = f"Boundary {objectCount}"
            if hasattr(object.manifold, "metadata"):
                name = object.manifold.metadata.get("Name", name)
            Interface_Static.SetCVal("write.step.product.name", name)
            for face in convert.convert_boundary_to_faces(object):
                writer.Transfer(face, STEPControl_AsIs)
        elif isinstance(object, Solid):
            shape = convert.convert_solid_to_shape(object)
            Interface_Static.SetCVal("write.step.product.name", f"Solid {objectCount}")
            writer.Transfer(shape, STEPControl_AsIs)
        elif isinstance(object, TopoDS_Shape):
            Interface_Static.SetCVal("write.step.product.name", f"Shape {objectCount}")
            writer.Transfer(object, STEPControl_AsIs)
        else:
            raise ValueError("Invalid object type")
        objectCount += 1

    # Create STEP header.
    model = writer.Model()
    model.ClearHeader()

    header = APIHeaderSection_MakeHeader()
    header.SetName(TCollection_HAsciiString("BSpyConvert"))
    header.SetAuthorValue(1, TCollection_HAsciiString("Eric Brechner"))
    header.SetAuthorisation(TCollection_HAsciiString("BSpyConvert (c) 2024"))

    description = Interface_HArray1OfHAsciiString(1, 1)
    description.SetValue(1, TCollection_HAsciiString("Objects produced by BSpy"))
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

    status = writer.Write(fileName)

    if status != IFSelect_RetDone:
        raise AssertionError("Write failed")

def export_stl(fileName, object, mode="ascii", linear_deflection = 0.9, angular_deflection = 0.5):
    if isinstance(object, Manifold):
        surface, flipNormal, transform = convert.convert_manifold_to_surface(object)
        shape = convert.convert_surface_to_face(surface, flipNormal)
    elif isinstance(object, Solid):
        shape = convert.convert_solid_to_shape(object)
    elif isinstance(object, TopoDS_Shape):
        shape = object
    else:
        raise ValueError("Invalid object type")

    write_stl_file(shape, fileName, mode, linear_deflection, angular_deflection)
