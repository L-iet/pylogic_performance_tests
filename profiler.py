import argparse
import importlib.util
import inspect
import os
import sys
import timeit
import cProfile
import pstats
import io
from types import FunctionType, MethodType, ModuleType
from typing import IO, Any, Callable, Protocol, TextIO
from pathlib import Path


TEST_FUNC_PREFIX = "t_"
SETUP_FUNC_PREFIX = "s_"
TEST_CLASS_PREFIX = "Test"

ROOT_DIR_NAME = "pylogic_performance_tests"


def is_test_function(name: str) -> bool:
    return name.startswith(TEST_FUNC_PREFIX)


def is_test_class(name: str) -> bool:
    return name.startswith(TEST_CLASS_PREFIX)


def get_setup_function_name(name: str) -> str:
    if is_test_function(name):
        return f"{SETUP_FUNC_PREFIX}{name[len(TEST_FUNC_PREFIX):]}"
    return f"{SETUP_FUNC_PREFIX}{name}"


def get_function_body(source: str) -> str:
    """Extracts the body of a function source code as a string."""
    # Remove the indentation and the function definition line
    lines = source.splitlines()
    indent_prefix: str = ""
    for c in lines[1]:
        if not c.isspace():
            break
        indent_prefix += c
    body = [line.removeprefix(indent_prefix) for line in lines[1:]]
    return "\n".join(body)


def get_testing_function_body(func: Callable) -> str:
    """Extracts the body of a testing function source code as a string."""
    source = inspect.getsource(func)
    if "return" in source:
        print()
        raise ValueError(
            "Function should not return anything, but it has a return statement.",
            source.splitlines()[0],
        )
    # perhaps check for yield as well?
    return get_function_body(source)


def parse_target(target_str: str) -> tuple[str, str | None, str | None]:
    if "::" in target_str:
        file_path, name, *rest = target_str.split("::")
        if rest:
            return os.path.abspath(file_path), name, rest[0]
    else:
        file_path, name = target_str, None
    return os.path.abspath(file_path), name, None


def load_module_from_path(file_path: Path):
    # get last occurence of ROOT_DIR_NAME in the path
    # raise ValueError if ROOT_DIR_NAME is not in the path
    reversed_file_path_parts = list(reversed(file_path.parts))
    root_dir_ind = (
        len(file_path.parts) - 1 - reversed_file_path_parts.index(ROOT_DIR_NAME)
    )

    module_name = ".".join(file_path.parts[root_dir_ind:])
    if module_name.endswith(".py"):
        module_name = module_name[:-3]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"Could not find module spec for {file_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None, f"Loader for {file_path} is None"
    spec.loader.exec_module(module)
    return module, vars(module)


class Profiler(Protocol):
    output_file: TextIO

    def __init__(self, output_file: TextIO = sys.stdout): ...
    def run(
        self,
        func: Callable,
        func_name: str,
        setup_func: Callable | None = None,
        setup_func_name: str | None = None,
        module_globals: dict[str, Any] = {},
    ) -> None:
        """
        Run the profiler on the given function and output to
        the output file.
        """
        ...

    def complete(self) -> None:
        """
        Finalize the profiling, if necessary.
        This is called after all functions have been profiled.
        """
        ...


class TimeitProfiler:
    def __init__(self, output_file: TextIO = sys.stdout):
        self.output_file = output_file
        self.timing_data: list[tuple[str, float]] = []

    def run(
        self,
        func: Callable,
        func_name: str,
        setup_func: Callable | None = None,
        setup_func_name: str | None = None,
        module_globals: dict[str, Any] = {},
        number=10000,
    ):
        setup_body = "pass"
        if setup_func:
            setup_body = get_testing_function_body(setup_func)
        func_body = get_testing_function_body(func)
        timer = timeit.Timer(func_body, setup=setup_body, globals=module_globals)
        try:
            res = timer.timeit(number=number) / number
        except Exception as e:
            print(e)
            res = float("inf")
        self.timing_data.append((func_name, res))
        self.output_file.write(
            f"[TIMEIT - {func_name}] Average time over {number} runs: {res:.6f} sec\n"
        )

    def complete(self):
        if self.timing_data:
            longest = sorted(self.timing_data, key=lambda x: x[1], reverse=True)
            total = sum(t for _, t in self.timing_data)
            if longest:
                self.output_file.write(f"\n[SUMMARY] Total time: {total:.6f} sec\n")
                self.output_file.write(
                    f"Longest function/class: {longest[0][0]} ({longest[0][1]:.6f} sec)\n"
                )


class CProfileProfiler:
    def __init__(
        self,
        output_file: TextIO = sys.stdout,
        sort_key: str | pstats.SortKey = pstats.SortKey.TIME,
    ):
        self.output_file = output_file
        self.sort_key = sort_key

    def run(
        self,
        func: Callable,
        func_name: str,
        setup_func: Callable | None = None,
        setup_func_name: str | None = None,
        module_globals: dict[str, Any] = {},
    ):
        # need to recreate the the profiler because it accumulates
        # data when you re enable after disabling
        self.pr = cProfile.Profile()

        func_body = get_testing_function_body(func)

        testing_locals = {}
        if setup_func:
            setup_body = get_testing_function_body(setup_func)
            exec(setup_body, module_globals, testing_locals)
        self.output_file.write(f"[CPROFILE - {func_name} starting]\n")
        self.pr.runctx(func_body, module_globals, testing_locals)
        ps = pstats.Stats(self.pr, stream=self.output_file).sort_stats(self.sort_key)
        ps.print_stats()

    def complete(self):
        self.output_file.write(f"[CPROFILE - complete]\n")


def get_all_test_functions_and_classes(module: ModuleType):
    funcs = [
        (name, obj)
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if is_test_function(name)
    ]
    classes = [
        (name, cls)
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if is_test_class(name)
    ]
    return funcs, classes


def profile_and_time(
    module: ModuleType,
    module_globals: dict[str, Any],
    name: str | None,
    subname: str | None,
    profiler_name: str,
    verbose: bool,
    file_output: str | None,
):
    output_lines = []
    timing_data = []

    output_file: TextIO
    if file_output:
        file_output = os.path.abspath(file_output)
        output_file = open(file_output, "w")
    else:
        output_file = sys.stdout

    profiler: Profiler
    if profiler_name == "cProfile":
        profiler = CProfileProfiler(output_file=output_file)
    elif profiler_name == "timeit":
        profiler = TimeitProfiler(output_file=output_file)
    else:
        raise ValueError(f"Unknown profiler: {profiler_name}")

    if name:
        if hasattr(module, name):
            obj = getattr(module, name)
            if inspect.isfunction(obj):
                setup_func_name = get_setup_function_name(name)
                setup_func: Callable | None = getattr(module, setup_func_name, None)
                profiler.run(obj, name, setup_func, setup_func_name, module_globals)
            elif inspect.isclass(obj):
                instance = obj()
                if subname:
                    methods: list[tuple[str, Callable]] = [
                        (subname, getattr(instance, subname))
                    ]
                    setup_name = get_setup_function_name(subname)
                    setup_func = getattr(instance, setup_name, None)
                    setup_methods = [(setup_name, setup_func)]
                else:
                    methods = inspect.getmembers(instance, predicate=inspect.ismethod)
                    setup_methods: list[tuple[str, Callable | None]] = []
                    for meth_name, meth in methods:
                        setup_meth_name = get_setup_function_name(meth_name)
                        setup_meth = getattr(instance, setup_meth_name, None)
                        setup_methods.append((setup_meth_name, setup_meth))
                for (meth_name, meth), (setup_meth_name, setup_meth) in zip(
                    methods, setup_methods
                ):
                    if is_test_function(meth_name):
                        profiler.run(
                            meth,
                            f"{name}.{meth_name}",
                            setup_meth,
                            f"{name}.{setup_meth_name}",
                            module_globals,
                        )
            else:
                print(f"Unsupported target type: {type(obj)}")
        else:
            print(f"Name '{name}' not found in module")
    else:
        funcs, classes = get_all_test_functions_and_classes(module)
        for func_name, func in funcs:
            setup_func_name = get_setup_function_name(func_name)
            setup_func: Callable | None = getattr(module, setup_func_name, None)
            profiler.run(func, func_name, setup_func, setup_func_name, module_globals)
        for class_name, cls in classes:
            instance = cls()
            methods = inspect.getmembers(instance, predicate=inspect.ismethod)
            setup_methods: list[tuple[str, Callable | None]] = []
            for meth_name, meth in methods:
                setup_meth_name = get_setup_function_name(meth_name)
                setup_meth = getattr(instance, setup_meth_name, None)
                setup_methods.append((setup_meth_name, setup_meth))
            for (meth_name, meth), (setup_meth_name, setup_meth) in zip(
                methods, setup_methods
            ):
                if is_test_function(meth_name):
                    profiler.run(
                        meth,
                        f"{class_name}.{meth_name}",
                        setup_meth,
                        f"{class_name}.{setup_meth_name}",
                        module_globals,
                    )

    profiler.complete()
    output_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Profile or time Python test functions/classes."
    )
    parser.add_argument(
        "target",
        help="Target in format path/to/file.py::function_name or path/to/file.py::TestClassName::method_name",
    )
    parser.add_argument(
        "-p",
        "--profiler",
        choices=["cProfile", "timeit"],
        default="cProfile",
        help="Profiler to use (default: cProfile)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-o", "--output-file", help="Path to save output to")

    args = parser.parse_args()

    file_path, name, subname = parse_target(args.target)
    if not os.path.isfile(file_path):
        print(f"Error: File not found - {file_path}")
        sys.exit(1)

    path = Path(file_path).resolve()
    module, module_globals = load_module_from_path(path)
    profile_and_time(
        module,
        module_globals,
        name,
        subname,
        args.profiler,
        args.verbose,
        args.output_file,
    )


if __name__ == "__main__":
    main()
