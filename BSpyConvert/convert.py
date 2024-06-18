import numpy as np
from collections import namedtuple
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_WireDone 
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.ShapeExtend import ShapeExtend_WireData
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve2d
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_IN
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

def convert_domain_to_wires(surface, domain):
    if domain.dimension != 2: raise ValueError("Domain must be 2D (dimension == 2)")
    if domain.containsInfinity: raise ValueError("Domain must be finite (containsInfinity == False)")

    # First, collect all manifold contour endpoints, accounting for slight numerical error.
    class Endpoint:
        def __init__(self, curve, t, clockwise, isStart, otherEnd=None):
            self.curve = curve
            self.t = t
            self.xy = curve.manifold.evaluate(t)
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
    wires = []
    holes = []
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
        useBuilder = True
        while next is not None:
            endpoints.remove(next)
            endpoints.remove(next.otherEnd)
            curve, rescale = convert_manifold_to_curve(next.curve.manifold)
            edge = BRepBuilderAPI_MakeEdge(curve, surface, rescale * next.t, rescale * next.otherEnd.t).Edge()
            wireData.Add(edge, wireData.NbEdges() + 1)
            if useBuilder:
                builder.Add(edge)
                if builder.Error() != BRepBuilderAPI_WireDone:
                    useBuilder = False
            next = next.otherEnd.connection
            if next is start:
                break
        if next is None:
            useBuilder = False
        if useBuilder and builder.IsDone():
            wire = builder.Wire()
        else:
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
            wire.Reverse()
        
        startUV = start.curve.manifold.evaluate(start.t)
        wires.append((gp_Pnt2d(float(startUV[0]), float(startUV[1])), wire))

    return wires

def convert_surface_to_face(surface, flipNormal = False):
    face = BRepBuilderAPI_MakeFace(surface, 1.0e-6).Face()
    if flipNormal:
        face.Reverse()
    return face

def convert_boundary_to_faces(boundary):
    surface, flipNormal, transform = convert_manifold_to_surface(boundary.manifold)
    domain = boundary.domain if transform is None else boundary.domain.transform(transform)
    wires = convert_domain_to_wires(surface, domain)

    # Build faces without holes.
    builders = []
    for (pnt2d, wire) in wires:
        builders.append(BRepBuilderAPI_MakeFace(surface, wire))
    
    # Determine which wires are within which other faces.
    classifier = BRepClass_FaceClassifier()
    nestings = []
    for i, (pnt2d, wire) in enumerate(wires):
        nesting = []
        for j, builder in enumerate(builders):
            if i != j:
                classifier.Perform(builder.Face(), pnt2d, Manifold.minSeparation)
                if classifier.State() == TopAbs_IN:
                    nesting.append(j)
        nestings.append(nesting)

    # Place holes within faces.
    for i, nesting in enumerate(nestings):
        if len(nesting) % 2 == 1:
            # The i'th wire is within an odd number of other faces.
            # Now, find the most deeply nested of those faces.
            j = nesting[0]
            maxDepth = len(nestings[j])
            for k in nesting[1:]:
                if maxDepth < len(nestings[k]):
                    j = k
                    maxDepth = len(nestings[k])
            # Remove this wire's face, since it's a hole in another face.
            builders[i] = None
            # Add hole to the other face.
            builders[j].Add(wires[i][1])

    # Create required 3D edges for remaining faces.
    faces = []
    for builder in builders:
        if builder is not None:
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
    explorer = TopologyExplorer(nurbs_shape, False)

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
            useFit = False
            weights = occSpline.Weights()
            if weights is None:
                order = (occSpline.UDegree() + 1, occSpline.VDegree() + 1)
                elevation = (0, 0)
                coefs = np.empty((3, occSpline.NbUPoles(), occSpline.NbVPoles()), float)
                for i in range(occSpline.NbUPoles()):
                    for j in range(occSpline.NbVPoles()):
                        pole = occSpline.Pole(i + 1, j + 1)
                        coefs[0, i, j] = pole.X()
                        coefs[1, i, j] = pole.Y()
                        coefs[2, i, j] = pole.Z()
            else:
                order = (max(occSpline.UDegree() + 1, 4 if occSpline.IsVRational() else 0),
                    max(occSpline.VDegree() + 1, 4 if occSpline.IsURational() else 0))
                elevation = (order[0] - occSpline.UDegree() - 1, order[1] - occSpline.VDegree() - 1)
                useFit = True
            knots = []
            unique = np.empty(occSpline.NbUKnots(), float)
            counts = np.empty(occSpline.NbUKnots(), int)
            for i in range(occSpline.NbUKnots()):
                unique[i] = occSpline.UKnot(i + 1)
                counts[i] = occSpline.UMultiplicity(i + 1) + elevation[0]
            if occSpline.IsUPeriodic():
                counts[0] = counts[-1] = order[0]
                for i, count in enumerate(counts[1:-1]):
                    counts[i + 1] = 0 if count < order[0] - 1 else count
                useFit = True
            knots.append(np.repeat(unique, counts))
            unique = np.empty(occSpline.NbVKnots(), float)
            counts = np.empty(occSpline.NbVKnots(), int)
            for i in range(occSpline.NbVKnots()):
                unique[i] = occSpline.VKnot(i + 1)
                counts[i] = occSpline.VMultiplicity(i + 1) + elevation[1]
            if occSpline.IsVPeriodic():
                counts[0] = counts[-1] = order[1]
                for i, count in enumerate(counts[1:-1]):
                    counts[i + 1] = 0 if count < order[1] - 1 else count
                useFit = True
            knots.append(np.repeat(unique, counts))

            # Create the Spline manifold.
            if useFit:
                # Fit the periodic and/or rational OCC surface.
                def evaluate_surface(uvw):
                    pnt = gp_Pnt()
                    occSpline.D0(uvw[0], uvw[1], pnt)
                    return np.array((pnt.X(), pnt.Y(), pnt.Z()))
                spline = Spline.fit(np.array(occSpline.Bounds()).reshape(2, 2), evaluate_surface, order, knots)
            else:
                spline = Spline(2, 3, order, coefs.shape[1:], knots, coefs)
            
            # Set proper orientation.
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
                useFit = False
                weights = occSpline.Weights()
                if weights is None:
                    order = occSpline.Degree() + 1
                    elevation = 0
                    coefs = np.empty((2, occSpline.NbPoles()), float)
                    for i in range(occSpline.NbPoles()):
                        pole = occSpline.Pole(i + 1)
                        coefs[0, i] = pole.X()
                        coefs[1, i] = pole.Y()
                else:
                    order = max(occSpline.Degree() + 1, 4 if occSpline.IsRational() else 0)
                    elevation = order - occSpline.Degree() - 1
                    useFit = True

                unique = np.empty(occSpline.NbKnots(), float)
                counts = np.empty(occSpline.NbKnots(), int)
                for i in range(occSpline.NbKnots()):
                    unique[i] = occSpline.Knot(i + 1)
                    counts[i] = occSpline.Multiplicity(i + 1) + elevation
                if occSpline.IsPeriodic():
                    counts[0] = counts[-1] = order
                    for i, count in enumerate(counts[1:-1]):
                        counts[i + 1] = 0 if count < order - 1 else count
                    useFit = True
                knots = np.repeat(unique, counts)

                # Create the domain spline manifold.
                if useFit:
                    # Fit the periodic and/or rational OCC curve.
                    def evaluate_curve(uvw):
                        pnt = gp_Pnt2d()
                        occSpline.D0(uvw[0], pnt)
                        return np.array((pnt.X(), pnt.Y()))
                    domainSpline = Spline.fit(np.array(((occSpline.FirstParameter(), occSpline.LastParameter()),)), evaluate_curve, (order,), (knots,))
                else:
                    domainSpline = Spline(1, 2, (order,), coefs.shape[1:], (knots,), coefs)

                # Set proper orientation.
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
