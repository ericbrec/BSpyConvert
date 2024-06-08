
##Copyright 2022 Thomas Paviot (tpaviot@gmail.com)
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

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.Interface import Interface_HArray1OfHAsciiString
from OCC.Core.APIHeaderSection import APIHeaderSection_MakeHeader
from OCC.Core.TCollection import TCollection_HAsciiString

import numpy as np
from bspy import Spline
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger

print(Geom_BSplineSurface.MaxDegree())

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

# initialize the STEP exporter
step_writer = STEPControl_Writer()
Interface_Static.SetCVal("write.step.schema", "AP203")

# transfer shapes and write file
Interface_Static.SetCVal("write.step.product.name", spline.metadata["Name"])
step_writer.Transfer(occSplineFace, STEPControl_AsIs)

#
# Set STEP header
#
model = step_writer.Model()
model.ClearHeader()

hs = APIHeaderSection_MakeHeader()
hs.SetName(TCollection_HAsciiString("BSpy Spline"))
hs.SetAuthorValue(1, TCollection_HAsciiString("Eric Brechner"))
hs.SetAuthorisation(TCollection_HAsciiString("BSpy (c) 2024"))

description = Interface_HArray1OfHAsciiString(1, 1)
description.SetValue(1, TCollection_HAsciiString("A b-spline surface produced by BSpy"))
hs.SetDescription(description)

org = Interface_HArray1OfHAsciiString(1, 1)
org.SetValue(1, TCollection_HAsciiString("BSpy organization"))
hs.SetOrganization(org)

hs.SetOriginatingSystem(TCollection_HAsciiString("BSpyConvert"))
#hs.SetImplementationLevel(TCollection_HAsciiString("implementation level"))

identifiers = Interface_HArray1OfHAsciiString(1, 1)
identifiers.SetValue(1, TCollection_HAsciiString("OpenCascade (pythonocc)"))
hs.SetSchemaIdentifiers(identifiers)

#hs.SetPreprocessorVersion(TCollection_HAsciiString("preprocessor version"))
#hs.SetTimeStamp(TCollection_HAsciiString(f"Time stamp: {datetime.now()}"))

model.AddHeaderEntity(hs.FnValue())
model.AddHeaderEntity(hs.FsValue())
model.AddHeaderEntity(hs.FdValue())

# finally write file
status = step_writer.Write(r"c:\users\ericb\onedrive\desktop\Surface1.stp")

if status != IFSelect_RetDone:
    raise AssertionError("load failed")