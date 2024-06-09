import numpy as np
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.GeomAbs import GeomAbs_BSplineSurface
from bspy import Spline, Hyperplane, Boundary, Solid

def convert_spline_to_surface(spline):
    if spline.nInd != 2: raise ValueError("Spline must be a surface (nInd == 2)")
    if spline.nDep != 3: raise ValueError("Spline must be a 3D surface (nDep == 3)")
    if spline.order[0] <= 1 or spline.order[1] <= 1: raise ValueError("Spline order must be greater than 1")
    if spline.order[0] > Geom_BSplineSurface.MaxDegree() or spline.order[1] > Geom_BSplineSurface.MaxDegree(): raise ValueError("Spline order must be <= Geom_BSplineSurface.MaxDegree")

    poles = TColgp_Array2OfPnt(1, spline.nCoef[0], 1, spline.nCoef[1])
    for i in range(spline.nCoef[0]):
        for j in range(spline.nCoef[1]):
            poles.SetValue(i + 1, j + 1, gp_Pnt(float(spline.coefs[0, i, j]), float(spline.coefs[1, i, j]), float(spline.coefs[2, i, j])))

    knots, multiplicity = np.unique(spline.knots[0], return_counts=True)
    uKnots = TColStd_Array1OfReal(1, len(knots))
    uMultiplicity = TColStd_Array1OfInteger(1, len(knots))
    for i in range(len(knots)):
        uKnots.SetValue(i + 1, float(knots[i]))
        uMultiplicity.SetValue(i + 1, int(multiplicity[i]))

    knots, multiplicity = np.unique(spline.knots[1], return_counts=True)
    vKnots = TColStd_Array1OfReal(1, len(knots))
    vMultiplicity = TColStd_Array1OfInteger(1, len(knots))
    for i in range(len(knots)):
        vKnots.SetValue(i + 1, float(knots[i]))
        vMultiplicity.SetValue(i + 1, int(multiplicity[i]))

    occSpline = Geom_BSplineSurface(poles, uKnots, vKnots, uMultiplicity, vMultiplicity, spline.order[0] - 1, spline.order[1] - 1)
    return occSpline

def convert_surface_to_face(surface, domain = None):
    if domain is None:
        return BRepBuilderAPI_MakeFace(surface, 1.0e-6).Face()
    
    return None

def convert_solid_to_shape(solid):
    if solid.dimension != 3: raise ValueError("Solid must be 3D (dimension == 3)")
    if solid.containsInfinity: raise ValueError("Solid must be finite (containsInfinity == False)")

def convert_shape_to_solid(shape):
    # Create empty solid.
    solid = Solid(3, False)

    # Convert all shape geometry to nurbs.
    nurbs_shape = BRepBuilderAPI_NurbsConvert(shape, True).Shape()

    # Now, all edges should be BSpline curves and surfaces BSpline surfaces.
    # See https://www.opencascade.com/doc/occt-7.4.0/refman/html/class_b_rep_builder_a_p_i___nurbs_convert.html#details

    # Convert each face to a Boundary with a Spline manifold.
    for face in TopologyExplorer(nurbs_shape).faces():
        surface = BRepAdaptor_Surface(face, True)
        if not surface.GetType() == GeomAbs_BSplineSurface:
            raise AssertionError("Face was not converted to a Geom_BSplineSurface")
        
        # Get the BSpline parameters.
        occSpline = surface.BSpline()
        order = (occSpline.UDegree() + 1, occSpline.VDegree() + 1)
        nCoef = (occSpline.NbUPoles(), occSpline.NbVPoles())
        knots = (np.empty(order[0] + nCoef[0], float), np.empty(order[1] + nCoef[1], float))
        uKnots = occSpline.UKnotSequence()
        for i in range(order[0] + nCoef[0]):
            knots[0][i] = uKnots.Value(i + 1)
        vKnots = occSpline.VKnotSequence()
        for i in range(order[1] + nCoef[1]):
            knots[1][i] = vKnots.Value(i + 1)
        poles = occSpline.Poles()
        weights = occSpline.Weights()
        nDep = 3 if weights is None else 4
        coefs = np.empty((nDep, nCoef[0], nCoef[1]), float)
        for i in range(nCoef[0]):
            for j in range(nCoef[1]):
                pole = poles.Value(i + 1, j + 1)
                coefs[0, i, j] = pole.X()
                coefs[1, i, j] = pole.Y()
                coefs[2, i, j] = pole.Z()
                if nDep > 3:
                    coefs[3, i, j] = weights.Value(i + 1, j + 1)

        # Create the Spline manifold and the Boundary, then add the boundary to the solid.
        spline = Spline(2, nDep, order, nCoef, knots, coefs)
        boundary = Boundary(spline, Hyperplane.create_hypercube(spline.domain()))
        solid.add_boundary(boundary)
    
    return solid
