##Copyright 2010-2017 Thomas Paviot (tpaviot@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>.

from OCC.Display.SimpleGui import init_display

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.GeomAbs import GeomAbs_BSplineSurface

import numpy as np
from bspy import Spline, Hyperplane, Boundary, Solid, Viewer

def convert_shape(shape):
    # Create empty solid.
    solid = Solid(3, False)

    # Convert all surfaces to nurbs
    nurbs_converter = BRepBuilderAPI_NurbsConvert(shape, True) # True means create copy
    converted_shape = nurbs_converter.Shape()

    # Now, all edges should be BSpline curves and surfaces BSpline surfaces.
    # See https://www.opencascade.com/doc/occt-7.4.0/refman/html/class_b_rep_builder_a_p_i___nurbs_convert.html#details

    # Loop over faces
    faceCount = 0
    for face in TopologyExplorer(converted_shape).faces():
        surface = BRepAdaptor_Surface(face, True)
        # check each of the is a BSpline surface
        # it should be, since we used the nurbs converter before
        if not surface.GetType() == GeomAbs_BSplineSurface:
            raise AssertionError(f"Face {faceCount} was not converted to a GeomAbs_BSplineSurface")
        # get the nurbs
        occSpline = surface.BSpline()
        order = (occSpline.UDegree() + 1, occSpline.VDegree() + 1)
        nCoef = (occSpline.NbUPoles(), occSpline.NbVPoles())
        knots = (np.empty(order[0] + nCoef[0], float), np.empty(order[1] + nCoef[1], float))
        # uKnots array
        uKnots = occSpline.UKnotSequence()
        for i in range(order[0] + nCoef[0]):
            knots[0][i] = uKnots.Value(i + 1)
        # vKnots array
        vKnots = occSpline.VKnotSequence()
        for i in range(order[1] + nCoef[1]):
            knots[1][i] = vKnots.Value(i + 1)
        # weights, a 2d array
        weights = occSpline.Weights()
        # weights can be None
        if False and weights is not None:
            print("Weights:", end="")
            for i in range(occSpline.NbUPoles()):
                for j in range(occSpline.NbVPoles()):
                    print(weights.Value(i + 1, j + 1), end=" ")
        # Coefficients (aka poles), as a 2d array
        poles = occSpline.Poles()
        coefs = np.empty((3, nCoef[0], nCoef[1]), float)
        for i in range(nCoef[0]):
            for j in range(nCoef[1]):
                pole = poles.Value(i + 1, j + 1)
                coefs[0, i, j] = pole.X()
                coefs[1, i, j] = pole.Y()
                coefs[2, i, j] = pole.Z()

        spline = Spline(2, 3, order, nCoef, knots, coefs)
        spline.metadata["Name"] = f"Boundary {faceCount}"
        boundary = Boundary(spline, Hyperplane.create_hypercube(spline.domain()))
        solid.add_boundary(boundary)
        faceCount += 1
    
    return solid

if __name__ == "__main__":
    viewer = Viewer()

    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(r"c:\users\ericb\onedrive\desktop\Surface1.stp")

    if status != IFSelect_RetDone:
        raise AssertionError("Error: can't read file.")
    transfer_result = step_reader.TransferRoots()
    if not transfer_result:
        raise AssertionError("Transfer failed.")
    for i in range(step_reader.NbShapes()):
        shape = step_reader.Shape(i + 1)
        if not shape.IsNull():
            solid = convert_shape(shape)
            viewer.draw(solid)
    
    viewer.mainloop()
