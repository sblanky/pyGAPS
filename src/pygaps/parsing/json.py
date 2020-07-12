"""Parse to and from a JSON file format for isotherms."""

import json

import pandas

from pygaps.core.isotherm import Isotherm
from pygaps.core.modelisotherm import ModelIsotherm
from pygaps.core.pointisotherm import PointIsotherm
from pygaps.modelling import model_from_dict
from pygaps.utilities.exceptions import ParsingError
from pygaps.utilities.unit_converter import _MASS_UNITS
from pygaps.utilities.unit_converter import _MOLAR_UNITS
from pygaps.utilities.unit_converter import _PRESSURE_UNITS
from pygaps.utilities.unit_converter import _VOLUME_UNITS


def isotherm_to_json(isotherm, path=None, **args_to_json):
    """
    Convert an isotherm object to a json representation.

    If the path is specified, the isotherm is saved as a file,
    otherwise it is returned as a string.
    Structure is inspired by the NIST format.

    Parameters
    ----------
    isotherm : Isotherm
        Isotherm to be written to json.
    path : str, None
        Path to the file to be written.
    args_to_json : dict
        Custom arguments to be passed to "json.dump".

    Returns
    -------
    None or str
        If path is None, returns the resulting json format as a string.
        Otherwise returns None.

    """
    # Isotherm properties
    raw_dict = isotherm.to_dict()

    # Isotherm data
    if isinstance(isotherm, PointIsotherm):

        # we turn the raw dataframe into a dictionary
        isotherm_data_dict = isotherm.data_raw.to_dict(orient='index')

        def process_data(value):
            """
            Specifically mark only the desorption branch.
            """
            if value.get('branch', False) is False:
                del value['branch']
            else:
                value['branch'] = 'des'
            return value

        raw_dict["isotherm_data"] = [
            process_data(v) for k, v in isotherm_data_dict.items()
        ]

    elif isinstance(isotherm, ModelIsotherm):
        raw_dict["isotherm_model"] = isotherm.model.to_dict()

    args_to_json = {} if args_to_json is None else args_to_json
    args_to_json['sort_keys'] = True  # we will sort always

    if path:
        with open(path, mode='w') as file:
            json.dump(raw_dict, file, **args_to_json)
    else:
        return json.dumps(raw_dict, **args_to_json)


def isotherm_from_json(
    str_or_path,
    fmt=None,
    loading_key='loading',
    pressure_key='pressure',
    **isotherm_parameters
):
    """
    Read a pyGAPS isotherm from a file or from a string.

    Structure is inspired by the NIST format.

    Parameters
    ----------
    str_or_path : str
        The isotherm in a json string format or a path
        to where one can be read.
    loading_key : str
        The title of the pressure data in the json provided.
    pressure_key
        The title of the loading data in the json provided.
    fmt : {None, 'NIST'}, optional
        If the format is set to NIST, then the json format a specific version
        used by the NIST database of adsorbents.

    Other Parameters
    ----------------
    isotherm_parameters :
        Any other options to be overridden in the isotherm creation.

    Returns
    -------
    Isotherm
        The isotherm contained in the json string or file.

    """
    # Parse isotherm in dictionary
    try:
        with open(str_or_path) as f:
            raw_dict = json.load(f)
    except OSError:
        try:
            raw_dict = json.loads(str_or_path)
        except Exception:
            raise ParsingError(
                "Could not parse JSON isotherm. "
                "The `str_or_path` is invalid or does not exist. "
            )

    # Update dictionary with any user parameters
    raw_dict.update(isotherm_parameters)

    data = raw_dict.pop("isotherm_data", None)
    model = raw_dict.pop("isotherm_model", None)

    if data:
        # rename keys and get units if needed depending on format
        if fmt == 'NIST':
            loading_key = 'total_adsorption'
            pressure_key = 'pressure'
            raw_dict = _from_json_nist(raw_dict)
            data = _from_data_nist(data)

        # build pandas dataframe of data
        data = pandas.DataFrame(data, dtype='float64')

        # process isotherm branches if they exist
        if 'branch' in data.columns:
            data['branch'] = data['branch'].fillna(False).replace('des', True)
        else:
            raw_dict['branch'] = 'guess'

        # get the other data in the json
        other_keys = [
            column for column in data.columns.values
            if column not in [loading_key, pressure_key, 'branch']
        ]

        # generate the isotherm
        isotherm = PointIsotherm(
            isotherm_data=data,
            loading_key=loading_key,
            pressure_key=pressure_key,
            other_keys=other_keys,
            **raw_dict
        )
    elif model:
        # generate the isotherm
        isotherm = ModelIsotherm(
            model=model_from_dict(model),
            **raw_dict,
        )

    else:
        # generate the isotherm
        isotherm = Isotherm(**raw_dict)

    return isotherm


def _from_json_nist(raw_dict):
    """Convert a NIST dictionary format to an internal format."""

    nist_dict = dict()

    # Get regular isotherm parameters
    nist_dict['material'] = raw_dict['adsorbent']['name']
    nist_dict['nist_hash'] = raw_dict.pop('adsorbent')['hashkey']
    nist_dict['temperature'] = raw_dict.pop('temperature')

    # Get adsorbate
    if len(raw_dict['adsorbates']) > 1:
        raise ParsingError('Cannot currently read multicomponent isotherms')
    nist_dict['adsorbate'] = raw_dict.pop('adsorbates')[0]['name'].lower()

    # Get loading basis and unit
    loading_string = raw_dict.pop("adsorptionUnits")
    comp = loading_string.split('/')
    if len(comp) != 2:
        if comp[0] == 'wt%':
            comp = ('g', 'g')
        else:
            raise ParsingError(
                "Isotherm cannot be parsed due to loading string format."
            )

    loading_unit = comp[0]
    if loading_unit in _MOLAR_UNITS:
        loading_basis = 'molar'
    elif loading_unit in _MASS_UNITS:
        loading_basis = 'mass'
    elif loading_unit in _VOLUME_UNITS:
        loading_basis = 'volume'
    else:
        raise ParsingError("Isotherm cannot be parsed due to loading unit.")

    adsorbent_unit = comp[1]
    if adsorbent_unit in _MASS_UNITS:
        adsorbent_basis = "mass"
    elif adsorbent_unit in _VOLUME_UNITS:
        adsorbent_basis = "volume"
    elif adsorbent_unit in _MOLAR_UNITS:
        adsorbent_basis = "molar"
    else:
        raise ParsingError("Isotherm cannot be parsed due to adsorbent basis.")

    # Get pressure mode and unit
    pressure_mode = "absolute"
    pressure_string = raw_dict.pop("pressureUnits")

    if pressure_string in _PRESSURE_UNITS:
        pressure_unit = pressure_string
    else:
        raise ParsingError("Isotherm cannot be parsed due to pressure unit.")

    # Add all the rest of the parameters
    nist_dict['iso_type'] = raw_dict.pop('category')  # exp/sim/mod/qua
    nist_dict['iso_ref'] = raw_dict.pop('isotherm_type')  # excess/absolute
    nist_dict.update(raw_dict)

    nist_dict.update({
        'adsorbent_basis': adsorbent_basis,
        'adsorbent_unit': adsorbent_unit,
        'loading_basis': loading_basis,
        'loading_unit': loading_unit,
        'pressure_mode': pressure_mode,
        'pressure_unit': pressure_unit,
    })

    return nist_dict


def _from_data_nist(data_raw):
    """Convert a NIST data format to an internal format."""

    for point in data_raw:
        point.pop('species_data')

    return data_raw