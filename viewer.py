#!/usr/bin/env python

##Copyright 2019 Thomas Paviot (tpaviot@gmail.com)
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

import numpy as np
from bspy import Spline
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.Graphic3d import Graphic3d_NOM_ALUMINIUM
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger

print(Geom_BSplineSurface.MaxDegree())

display, start_display, add_menu, add_function_to_menu = init_display()

[spline] = Spline.load(r"c:\users\ericb\onedrive\desktop\Surface1.json")

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
occSplineFace = BRepBuilderAPI_MakeFace(occSpline, 1.0e-6).Face()
color = spline.metadata["fillColor"]
display.DisplayShape(occSplineFace, material=Graphic3d_NOM_ALUMINIUM, color=Quantity_Color(float(color[0]), float(color[1]), float(color[2]), Quantity_TOC_RGB))

display.FitAll()
start_display()
