import numpy as np
from collections import namedtuple
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
from OCC.Core.ShapeExtend import ShapeExtend_WireData
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve2d
from OCC.Core.TopAbs import TopAbs_FORWARD
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Geom import Geom_BSplineSurface, Geom_Plane
from OCC.Core.Geom2d import Geom2d_BSplineCurve, Geom2d_Line
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pnt2d, gp_Dir2d
from OCC.Core.TColgp import TColgp_Array2OfPnt, TColgp_Array1OfPnt2d
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.GeomAbs import GeomAbs_BSplineSurface, GeomAbs_BSplineCurve
from bspy import Manifold, Spline, Hyperplane, Boundary, Solid

def convert_manifold_to_surface(manifold):
    if manifold.range_dimension() != 3: raise ValueError("Manifold must be a 3D surface")

    if isinstance(manifold, Hyperplane):
        hyperplane = manifold
        point = gp_Pnt(float(hyperplane._point[0]), float(hyperplane._point[1]), float(hyperplane._point[2]))
        normal = gp_Dir(float(hyperplane._normal[0]), float(hyperplane._normal[1]), float(hyperplane._normal[2]))
        xAxis = gp_Dir(float(hyperplane._tangentSpace[0, 0]), float(hyperplane._tangentSpace[1, 0]), float(hyperplane._tangentSpace[2, 0]))
        axes = gp_Ax3(point, normal, xAxis)
        surface = Geom_Plane(axes)
        flipNormal = False
        xDirection = axes.XDirection()
        yDirection = axes.YDirection()
        tangentSpace = np.array(((xDirection.X(), xDirection.Y(), xDirection.Z()),
            (yDirection.X(), yDirection.Y(), yDirection.Z()))).T
        transform = np.linalg.inv(tangentSpace.T @ tangentSpace) @ (tangentSpace.T @ hyperplane._tangentSpace)
    elif isinstance(manifold, Spline):
        spline = manifold
        if spline.nInd != 2: raise ValueError("Spline must be a surface (nInd == 2)")
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

        surface = Geom_BSplineSurface(poles, uKnots, vKnots, uMultiplicity, vMultiplicity, spline.order[0] - 1, spline.order[1] - 1)
        flipNormal = spline.metadata.get("flipNormal", False)
        transform = None
    else:
        raise ValueError("Manifold must be a plane or spline")
    
    return surface, flipNormal, transform

def convert_manifold_to_curve(manifold):
    if manifold.range_dimension() != 2: raise ValueError("Manifold must be a 2D line or spline")

    if isinstance(manifold, Hyperplane):
        hyperplane = manifold
        point = gp_Pnt2d(float(hyperplane._point[0]), float(hyperplane._point[1]))
        vector = gp_Dir2d(float(hyperplane._tangentSpace[0, 0]), float(hyperplane._tangentSpace[1, 0]))
        curve = Geom2d_Line(point, vector)
        rescale = np.linalg.norm(hyperplane._tangentSpace[:, 0])
    elif isinstance(manifold, Spline):
        spline = manifold
        if spline.nInd != 1: raise ValueError("Spline must be a curve (nInd == 1)")
        if spline.nDep != 2: raise ValueError("Spline must be a 2D curve (nDep == 2)")
        if spline.order[0] <= 1: raise ValueError("Spline order must be greater than 1")
        if spline.order[0] > Geom2d_BSplineCurve.MaxDegree(): raise ValueError("Spline order must be <= Geom_BSplineSurface.MaxDegree")

        poles = TColgp_Array1OfPnt2d(1, spline.nCoef[0])
        for i in range(spline.nCoef[0]):
            poles.SetValue(i + 1, gp_Pnt2d(float(spline.coefs[0, i]), float(spline.coefs[1, i])))

        knots, multiplicity = np.unique(spline.knots[0], return_counts=True)
        uKnots = TColStd_Array1OfReal(1, len(knots))
        uMultiplicity = TColStd_Array1OfInteger(1, len(knots))
        for i in range(len(knots)):
            uKnots.SetValue(i + 1, float(knots[i]))
            uMultiplicity.SetValue(i + 1, int(multiplicity[i]))

        curve = Geom2d_BSplineCurve(poles, uKnots, uMultiplicity, spline.order[0] - 1)
        rescale = 1.0
    else:
        raise ValueError("Manifold must be a line or spline")
    
    return curve, rescale

def convert_domain_to_bundles(surface, domain):
    if domain.dimension != 2: raise ValueError("Domain must be 2D (dimension == 2)")
    if domain.containsInfinity: raise ValueError("Domain must be finite (containsInfinity == False)")

    # First, collect all manifold contour endpoints, accounting for slight numerical error.
    class Endpoint:
        def __init__(self, curve, t, clockwise, isStart, otherEnd=None):
            self.curve = curve
            self.t = t
            self.xy = curve.manifold.evaluate((t,))
            self.clockwise = clockwise
            self.isStart = isStart
            self.otherEnd = otherEnd
            self.connection = None
    endpoints = []
    for curve in domain.boundaries:
        curve.domain.boundaries.sort(key=lambda boundary: (boundary.manifold.evaluate(0.0), -boundary.manifold.normal(0.0)))
        leftB = 0
        rightB = 0
        boundaryCount = len(curve.domain.boundaries)
        while leftB < boundaryCount:
            if curve.domain.boundaries[leftB].manifold.normal(0.0) < 0.0:
                leftPoint = curve.domain.boundaries[leftB].manifold.evaluate(0.0)[0]
                while rightB < boundaryCount:
                    rightPoint = curve.domain.boundaries[rightB].manifold.evaluate(0.0)[0]
                    if leftPoint - Manifold.minSeparation < rightPoint and curve.domain.boundaries[rightB].manifold.normal(0.0) > 0.0:
                        t = curve.manifold.tangent_space(leftPoint)[:,0]
                        n = curve.manifold.normal(leftPoint)
                        clockwise = t[0] * n[1] - t[1] * n[0] > 0.0
                        ep1 = Endpoint(curve, leftPoint, clockwise, rightPoint >= leftPoint)
                        ep2 = Endpoint(curve, rightPoint, clockwise, rightPoint < leftPoint, ep1)
                        ep1.otherEnd = ep2
                        endpoints.append(ep1)
                        endpoints.append(ep2)
                        leftB = rightB
                        rightB += 1
                        break
                    rightB += 1
            leftB += 1

    # Second, collect all valid pairings of endpoints (normal not flipped between segments).
    Connection = namedtuple('Connection', ('distance', 'ep1', 'ep2'))
    connections = []
    for i, ep1 in enumerate(endpoints[:-1]):
        for ep2 in endpoints[i+1:]:
            if (ep1.clockwise == ep2.clockwise and ep1.isStart != ep2.isStart) or \
                (ep1.clockwise != ep2.clockwise and ep1.isStart == ep2.isStart):
                connections.append(Connection(np.linalg.norm(ep1.xy - ep2.xy), ep1, ep2))

    # Third, only keep closest pairings (prune the rest).
    connections.sort(key=lambda connection: -connection.distance)
    while connections:
        connection = connections.pop()
        connection.ep1.connection = connection.ep2
        connection.ep2.connection = connection.ep1
        connections = [c for c in connections if c.ep1 is not connection.ep1 and c.ep1 is not connection.ep2 and \
            c.ep2 is not connection.ep1 and c.ep2 is not connection.ep2]
        
    # Fourth, trace the contours from pairing to pairing.
    bundles = []
    while endpoints:
        start = endpoints[0]
        if not start.isStart:
            start = start.otherEnd
        # Run backwards until you hit start again or hit an end.
        if start.connection is not None:
            originalStart = start
            next = start.connection
            start = None
            while next is not None and start is not originalStart:
                start = next.otherEnd
                next = start.connection
        # Run forwards building the wire.
        next = start
        wireData = ShapeExtend_WireData()
        builder = BRepBuilderAPI_MakeWire()
        while next is not None:
            endpoints.remove(next)
            endpoints.remove(next.otherEnd)
            curve, rescale = convert_manifold_to_curve(next.curve.manifold)
            edge = BRepBuilderAPI_MakeEdge(curve, surface, rescale * next.t, rescale * next.otherEnd.t).Edge()
            wireData.Add(edge, wireData.NbEdges() + 1)
            builder.Add(edge)
            next = next.otherEnd.connection
            if next is start:
                break
        if next is None:
            print("edge not closed")
        if builder.IsDone():
            print("builder wire")
            wire = builder.Wire()
        else:
            print("fixer wire")
            fixer = ShapeFix_Wire()
            fixer.SetSurface(surface)
            fixer.SetPrecision(Manifold.minSeparation)
            fixer.SetPreferencePCurveMode(True)
            fixer.SetClosedWireMode(True)
            fixer.Load(wireData)
            fixer.Perform()
            wire = fixer.Wire()

        # Reverse the direction of the wire if its movement is clockwise.
        # The movement is clockwise if the start point moves clockwise (or its a counterclockwise ending point).
        if start.clockwise == start.isStart:
            print("Reverse")
            wire.Reverse()
        bundles.append([wire])

    return bundles

def convert_surface_to_face(surface, flipNormal = False):
    face = BRepBuilderAPI_MakeFace(surface, 1.0e-6).Face()
    if flipNormal:
        face.Reverse()
    return face

def convert_boundary_to_faces(boundary):
    surface, flipNormal, transform = convert_manifold_to_surface(boundary.manifold)
    domain = boundary.domain if transform is None else boundary.domain.transform(transform)
    bundles = convert_domain_to_bundles(surface, domain)

    faces = []
    for bundle in bundles:
        builder = BRepBuilderAPI_MakeFace(surface, bundle[0])
        for wire in bundle[1:]:
            builder.Add(wire)

        # Create required 3D edges
        fixer = ShapeFix_Face(builder.Face())
        fixer.Perform()
        faces.append(fixer.Face())
    
    return faces

def convert_solid_to_shape(solid):
    if solid.dimension != 3: raise ValueError("Solid must be 3D (dimension == 3)")
    if solid.containsInfinity: raise ValueError("Solid must be finite (containsInfinity == False)")

    builder = BRepBuilderAPI_Sewing(Manifold.minSeparation)
    for boundary in solid.boundaries:
        for face in convert_boundary_to_faces(boundary):
            builder.Add(face)
    
    builder.Perform()
    return builder.SewedShape()

def convert_shape_to_solid(shape):
    # Create empty solid.
    solid = Solid(3, False)

    # Convert all shape geometry to nurbs.
    nurbs_shape = BRepBuilderAPI_NurbsConvert(shape, True).Shape()
    explorer = TopologyExplorer(nurbs_shape)

    # Now, all edges should be BSpline curves and surfaces BSpline surfaces.
    # See https://www.opencascade.com/doc/occt-7.4.0/refman/html/class_b_rep_builder_a_p_i___nurbs_convert.html#details

    # Convert each face to a Boundary with a Spline manifold.
    for shell in explorer.shells():
        shellFlipped = shell.Orientation() != TopAbs_FORWARD
        
        for face in explorer.faces_from_solids(shell):
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

            # Create the Spline manifold.
            spline = Spline(2, nDep, order, nCoef, knots, coefs)
            faceFlipped = face.Orientation() != TopAbs_FORWARD
            faceFlipped = not faceFlipped if shellFlipped else faceFlipped
            if faceFlipped:
                spline = spline.flip_normal()

            # Create the spline domain boundaries.
            domain = Solid(2, False)
            for edge in explorer.edges_from_face(face):
                curve = BRepAdaptor_Curve2d(edge, face)
                if not curve.GetType() == GeomAbs_BSplineCurve:
                    raise AssertionError("Edge was not converted to a Geom_BSplineCurve")
                
                # Get the BSpline parameters.
                occSpline = curve.BSpline()
                order = (occSpline.Degree() + 1,)
                nCoef = (occSpline.NbPoles(),)
                knots = (np.empty(order[0] + nCoef[0], float),)
                uKnots = occSpline.KnotSequence()
                for i in range(order[0] + nCoef[0]):
                    knots[0][i] = uKnots.Value(i + 1)
                poles = occSpline.Poles()
                weights = occSpline.Weights()
                nDep = 2 if weights is None else 3
                coefs = np.empty((nDep, nCoef[0]), float)
                for i in range(nCoef[0]):
                    pole = poles.Value(i + 1)
                    coefs[0, i] = pole.X()
                    coefs[1, i] = pole.Y()
                    if nDep > 3:
                        coefs[2, i] = weights.Value(i + 1)

                # Create the domain spline manifold.
                domainSpline = Spline(1, nDep, order, nCoef, knots, coefs)
                edgeFlipped = edge.Orientation() != TopAbs_FORWARD
                edgeFlipped = not edgeFlipped if faceFlipped else edgeFlipped
                if edgeFlipped:
                    domainSpline = domainSpline.flip_normal()

                # Create the domain spline domain boundaries.
                domainSplineDomain = Solid(1, False)
                for vertex in explorer.vertices_from_edge(edge):
                    done, parameter = BRep_Tool.Parameter(vertex, edge)
                    if done:
                        vertexFlipped = vertex.Orientation() != TopAbs_FORWARD
                        vertexFlipped = not vertexFlipped if edgeFlipped else vertexFlipped
                        normal = 1.0 if vertexFlipped else -1.0
                        domainSplineDomain.add_boundary(Boundary(Hyperplane(normal, parameter, 0.0), Solid(0, True)))
                domain.add_boundary(Boundary(domainSpline, domainSplineDomain))

            # Create the solid boundary
            solid.add_boundary(Boundary(spline, domain))
    
    return solid
