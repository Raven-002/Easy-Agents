from typing import TypeAlias, Union

RecursiveStrDict: TypeAlias = dict[str, Union[str, "RecursiveStrDict"]]
