"""Class for command line input handling."""

from typing import Any
from typing import NoReturn
from typing import List
from typing import Optional

import click
import sys
import time
from threading import Thread

# Exit command line input
DEFAULT_INPUT = "EXIT"


class InputHandler:
    """General command line interface module.

    Raises:
        TimeoutError: Timeout for input reached.
        ValueError: Input is not valid.
    """

    @staticmethod
    def expect_input(
        prompt: str,
        choices: Optional[List[Any]] = [],
        confirm: Optional[bool] = False,
        timeout: Optional[int] = 30,  # default input time: 30 sec
    ) -> Any:
        """Outputs prompt to command line and expects an user input.

        Args:
            prompt (str): Command line output to describe the expected input.
            choices (Optional[List[Any]], optional): Options to choose from
            by command line input. Defaults to [].
            confirm (Optional[bool], optional): Flag if input should be confirmed.
            Defaults to False.
            timeout (Optional[int], optional): Time until program is quitted. Defaults to 30.

        Raises:
            TimeoutError: Time for input is up.
        """
        input_statement = None
        choices += [DEFAULT_INPUT]

        def set_timeout(limit: int = timeout) -> None:
            time.sleep(limit)
            if input_statement != None:
                return
            raise TimeoutError("Input timeout!")

        if not all([isinstance(choice, str) for choice in choices]):
            choices = list(map(str, choices))
        reloop = True

        # expect input until:
        # - input is confirmed
        # - input is given
        while reloop:
            thread = Thread(
                target=set_timeout,
            )
            thread.start()
            input_statement = click.prompt(
                prompt,
                default=DEFAULT_INPUT,
                show_default=True,
                type=click.Choice(choices) if choices else None,
                show_choices=True if choices else False,
            )
            if input_statement == DEFAULT_INPUT:
                sys.exit()
            if confirm:
                if click.confirm(f"Input: '{input_statement}' correct?"):
                    reloop = False
            else:
                reloop = False

        return input_statement

    @staticmethod
    def check_input(input_statement: Any, valid_inputs: List[str]) -> NoReturn:
        """Check if the command line input is valid.
        Can be used to trigger a reloop for the command line input.

        Args:
            input_statement (Any): Statement that is checked.
            valid_inputs (List[str]): List of valid objects as inputs.

        Raises:
            ValueError: Invalid input is given.
        """
        if not isinstance(input_statement, list):
            input_statement = [input_statement]
        invalid_inputs = []
        for item in input_statement:
            if not isinstance(item, str):
                item = str(item)
            if item not in valid_inputs:
                invalid_inputs.append(item)
        if invalid_inputs:
            raise ValueError("'%s' are invalid inputs!", str(invalid_inputs))
