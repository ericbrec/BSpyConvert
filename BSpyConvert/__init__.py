"""
BSpyConvert is a library for converting BSpy splines and solid models to and from OpenCascade (OCCT) equivalents and a variety of geometry and CAD file formats.

Available subpackages
---------------------
`bspy.convert` : Functions to convert BSpy objects to OpenCascade (OCCT) objects (and vice-versa).

`bspy.importer` : Functions to import BSpy objects from IGES and STEP files.

`bspy.exporter` : Functions to export BSpy objects to IGES, STEP, and STL files.
"""
from BSpyConvert.convert import convert_manifold_to_curve, convert_domain_to_wires, convert_manifold_to_surface, convert_surface_to_face, convert_boundary_to_faces, convert_solid_to_shape, convert_shape_to_solids
from BSpyConvert.importer import import_iges, import_step
from BSpyConvert.exporter import export_iges, export_step, export_stl
