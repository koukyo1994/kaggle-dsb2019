import ast
import importlib
import importlib.util as util
import sys

from typing import List

USERDEFINE = 0
OTHERS = 1


def origin_checker(spec: importlib.machinery.ModuleSpec,
                   prefix: str = "/app") -> int:
    origin: str = spec.origin  # type: ignore
    if origin.startswith(prefix):
        return USERDEFINE
    return OTHERS


def import_collector(pytree: ast.Module):
    modules_list: List[importlib.machinery.ModuleSpec] = []
    # from_modules_list: List[importlib.machinery.ModuleSpec] = []
    for node in pytree.body:
        if isinstance(node, ast.Import):
            for name in node.names:
                spec = util.find_spec(name.name)
                if spec is not None:
                    modules_list.append(spec)
        elif isinstance(node, ast.ImportFrom):
            pass
    return modules_list


if __name__ == "__main__":
    sys.path.append("./")

    pyfile = sys.argv[1]

    with open(pyfile, mode="r", encoding="utf-8") as f:
        pysrc = f.read()

    pytree = ast.parse(pysrc)
    modules_list = import_collector(pytree)

    import pdb
    pdb.set_trace()
