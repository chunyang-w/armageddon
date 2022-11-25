import numpy as np
from pytest import fixture


@fixture(scope='module')
def armageddon():
    import armageddon
    return armageddon


@fixture(scope='module')
def loc(armageddon):
    return armageddon.PostcodeLocator()


def test_import(armageddon):
    assert armageddon


def test_get_population(loc):

    postcode1 = [['AL1  2']]
    postcode2 = [['SW7  2']]
    postcode3 = [['SW7 2AZ', 'SW7 2BT', 'SW7 2BU', 'SW7 2DD']]

    sec1 = True
    sec2 = True
    sec3 = False

    result1 = loc.get_population_of_postcode(postcode1, sector=sec1)
    result2 = loc.get_population_of_postcode(postcode2, sector=sec2)
    result3 = loc.get_population_of_postcode(postcode3, sector=sec3)

    output1 = [[6523.0]]
    output2 = [[2283.0]]
    output3 = [[19.0, 19.0, 19.0, 19.0]]

    assert type(result1) is list
    assert type(result2) is list
    assert type(result3) is list

    if len(result1) > 0:
        for element in result1:
            assert type(element) is list

    if len(result2) > 0:
        for element in result2:
            assert type(element) is list

    if len(result3) > 0:
        for element in result3:
            assert type(element) is list

    assert np.allclose(result1, output1)
    assert np.allclose(result2, output2)
    assert np.allclose(result3, output3)
