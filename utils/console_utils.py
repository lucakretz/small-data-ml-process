"""Utility functions to exploit data to the command line interface in a nice format."""

from typing import List
from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Union
from typing import Any

from itertools import zip_longest
from tabulate import tabulate


def show_data(
    data_dict: Dict[str, Union[List[str], int]],
    header: List[Union[str, int]],
    key2row: Optional[bool] = True,
    fill_value: Optional[str] = "",
) -> NoReturn:
    """Displays inputs to command line.

    Args:
        data_dict (Dict[str, Union[List[str], int]]): Provides information
        to display.
        header (List[Union[str, int]]): Provides header for the output table.
        key2row (Optional[bool], optional): Switch if True the dict keys are
        added as an additional column. Defaults to True.
        fill_value (Optional[str], optional): String to fill empty rows.
        Defaults to "".
    """
    table = prepare_table(data_dict, fill_value=fill_value, key2row=key2row)
    print_table(table, header)


def print_table(table: List[List[str]], header: List[str]) -> NoReturn:
    """Prints tablulate to command line.

    Args:
        table (List[List[str]]): Information in table format.
        header (List[str]): Column names for table.
    """
    print("\n" + tabulate(table, headers=header))


def prepare_table(
    content_dict: Dict[str, Union[List[str], int]],
    fill_value: str,
    key2row: Optional[bool] = True,
) -> List[List[Any]]:
    """Prepares the data for the command line output.

    Args:
        content_dict (Dict[str, Union[List[str], int]]): Content
        that should be displayed.
        fill_value (str): Filler string for empty rows.
        key2row (Optional[bool], optional): Add keys as column or not.
        Defaults to True.

    Returns:
        List[List[Any]]: Data in table format.
    """
    if key2row:
        return transform_horizontal(content_dict)
    return transform_vertical(content_dict, fill_value)


def transform_vertical(
    dictionary: Dict[str, Union[List[str], int]], fill_value: str
) -> List[List[Any]]:
    """Produces rows of the upcoming table.

    Args:
        dictionary (Dict[str, Union[List[str], int]]): Row content.
        fill_value (str): String to fill up empty rows.

    Returns:
        List[List[Any]]: List of rows.
    """
    values = [list(value) for value in dictionary.values()]
    rows = list(zip_longest(*values, fillvalue=fill_value))
    return rows


def transform_horizontal(
    dictionary: Dict[str, Union[List[str], int]]
) -> List[List[Any]]:
    """Added keys to rows.

    Args:
        dictionary (Dict[str, Union[List[str], int]]): Content
        to display.

    Returns:
        List[List[Any]]: Rows with keys as first column.
    """
    rows = [[key, value] for key, value in dictionary.items()]
    return rows
