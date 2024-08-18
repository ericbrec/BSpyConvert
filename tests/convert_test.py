import BSpyConvert.convert as convert
import bspy

def test_island():
    epsilon = 5.0e-2

    [originalSolid] = bspy.Solid.load("tests/island.json")
    shape = convert.convert_solid_to_shape(originalSolid)
    convertedSolids = convert.convert_shape_to_solids(shape)

    computedVolume = 0.0
    for solid in convertedSolids:
        computedVolume += solid.volume_integral(lambda x: 1.0)
    expectedVolume = originalSolid.volume_integral(lambda x: 1.0)
    print(computedVolume, expectedVolume, abs(computedVolume - expectedVolume) / expectedVolume)
    assert abs(computedVolume - expectedVolume) < epsilon * expectedVolume

    computedSurfaceArea = 0.0
    for solid in convertedSolids:
        computedSurfaceArea += solid.surface_integral(lambda x, n: n)
    expectedSurfaceArea = originalSolid.surface_integral(lambda x, n: n)
    print(computedSurfaceArea, expectedSurfaceArea, abs(computedSurfaceArea - expectedSurfaceArea) / expectedSurfaceArea)
    assert abs(computedSurfaceArea - expectedSurfaceArea) < epsilon * expectedSurfaceArea

def test_teapots():
    epsilon = 5.0e-2

    [originalSolid] = bspy.Solid.load("tests/teapots.json")
    shape = convert.convert_solid_to_shape(originalSolid)
    convertedSolids = convert.convert_shape_to_solids(shape)

    computedVolume = 0.0
    for solid in convertedSolids:
        computedVolume += solid.volume_integral(lambda x: 1.0)
    expectedVolume = originalSolid.volume_integral(lambda x: 1.0)
    print(computedVolume, expectedVolume, abs(computedVolume - expectedVolume) / expectedVolume)
    assert abs(computedVolume - expectedVolume) < epsilon * expectedVolume

    computedSurfaceArea = 0.0
    for solid in convertedSolids:
        computedSurfaceArea += solid.surface_integral(lambda x, n: n)
    expectedSurfaceArea = originalSolid.surface_integral(lambda x, n: n)
    print(computedSurfaceArea, expectedSurfaceArea, abs(computedSurfaceArea - expectedSurfaceArea) / expectedSurfaceArea)
    assert abs(computedSurfaceArea - expectedSurfaceArea) < epsilon * expectedSurfaceArea