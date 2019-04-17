"""Tests excel parsing."""

import pytest

import pygaps

from .conftest import DATA_EXCEL_BEL
from .conftest import DATA_EXCEL_MIC
from .conftest import DATA_EXCEL_STD
from .conftest import DATA_JSON_STD


@pytest.mark.parsing
class TestExcel():

    def test_read_create_excel(self, basic_pointisotherm, tmpdir_factory):
        """Test creation of the regular excel file."""
        path = tmpdir_factory.mktemp('excel').join('regular.xls').strpath

        pygaps.isotherm_to_xl(basic_pointisotherm, path=path)

        isotherm = pygaps.isotherm_from_xl(path)
        assert isotherm == basic_pointisotherm

    def test_read_excel(self, basic_pointisotherm, tmpdir_factory):
        """Test read excel files file."""

        for index, path in enumerate(DATA_EXCEL_STD):
            isotherm = pygaps.isotherm_from_xl(path=path)
            with open(DATA_JSON_STD[index], 'r') as file:
                isotherm2 = pygaps.isotherm_from_json(file.read())
                assert isotherm.to_dict() == isotherm2.to_dict()

    def test_read_excel_mic(self):
        """Test reading of micromeritics report files."""

        for path in DATA_EXCEL_MIC:
            isotherm = pygaps.isotherm_from_xl(path=path, fmt='mic')
            json_path = path.replace('.xls', '.json')
            with open(json_path, 'r') as file:
                assert isotherm == pygaps.isotherm_from_json(file.read())

    def test_read_excel_bel(self):
        """Test reading of bel report files."""

        for path in DATA_EXCEL_BEL:
            isotherm = pygaps.isotherm_from_xl(path=path, fmt='bel')
            json_path = path.replace('.xls', '.json')
            with open(json_path, 'r') as file:
                new_iso = pygaps.isotherm_from_json(file.read())
                assert isotherm == new_iso
