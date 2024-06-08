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
from OCC.Extend.DataExchange import read_step_file

if __name__ == "__main__":
    display, start_display, add_menu, add_function_to_menu = init_display()

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
            display.DisplayShape(shape)
    display.FitAll()
    start_display()