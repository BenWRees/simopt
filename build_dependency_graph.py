#!/usr/bin/env python3

"""Build dependency graph utilities."""

import ast
import os
import sys
from collections import defaultdict, deque


def find_python_files(base_dir: str) -> list:
    """Find python files."""
    py_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.relpath(os.path.join(root, f), base_dir))  # noqa: PTH118
    return py_files


def module_name(file_path: str) -> str:
    """Module name."""
    return file_path[:-3].replace(os.sep, ".")


def parse_imports(file_path: str) -> set:
    """Parse imports."""
    with open(file_path, encoding="utf-8") as f:  # noqa: PTH123
        tree = ast.parse(f.read(), filename=file_path)

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])

        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    return imports


def build_graph(files: list) -> tuple:
    """Build graph."""
    {module_name(f): f for f in files}
    name_to_file = {os.path.basename(f)[:-3]: f for f in files}  # noqa: PTH119

    graph = defaultdict(set)
    in_degree = dict.fromkeys(files, 0)

    for f in files:
        imports = parse_imports(f)

        for imp in imports:
            dep_file = name_to_file.get(imp)

            if dep_file and dep_file != f and f not in graph[dep_file]:
                graph[dep_file].add(f)
                in_degree[f] += 1

    return graph, in_degree


def topo_layers(graph: defaultdict(set), in_degree: dict) -> list:  # type: ignore
    """Topo layers."""
    queue = deque([f for f in in_degree if in_degree[f] == 0])
    layers = []

    while queue:
        current_layer = list(queue)
        layers.append(current_layer)
        next_queue = deque()

        for node in current_layer:
            for neigh in graph[node]:
                in_degree[neigh] -= 1
                if in_degree[neigh] == 0:
                    next_queue.append(neigh)

        queue = next_queue

    # Detect cycles
    if any(v > 0 for v in in_degree.values()):
        print("WARNING: Cycle detected in dependencies", file=sys.stderr)

    return layers


def main() -> None:
    """Run main entry point."""
    base_dir = "."

    files = find_python_files(base_dir)

    # Optional: restrict to modified files passed via stdin
    if not sys.stdin.isatty():
        modified = {line.strip() for line in sys.stdin if line.strip()}
        files = [f for f in files if f in modified]

    graph, in_degree = build_graph(files)
    layers = topo_layers(graph, in_degree)

    # Output format: one layer per line
    for layer in layers:
        print(" ".join(layer))


if __name__ == "__main__":
    main()
