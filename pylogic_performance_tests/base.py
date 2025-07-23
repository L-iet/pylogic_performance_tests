from typing import Self
from pylogic.base import _PylogicObject, string_match, matches_to_actual
import copy


class FakePylogicObject(_PylogicObject):
    child_independent_attrs = _PylogicObject.child_independent_attrs + ("name",)
    child_dependent_attrs = _PylogicObject.child_dependent_attrs
    hash_attrs = _PylogicObject.hash_attrs + ("name",)
    __slots__ = child_dependent_attrs + child_independent_attrs

    def __init__(
        self,
        name: str | None = None,
        children: list[_PylogicObject] | None = None,
        **kwargs,
    ):
        super().__init__(children=children, name=name, **kwargs)

    def update_child_dependent_attrs(self) -> None:
        return super().update_child_dependent_attrs()

    def init_child_independent_attrs(self, name: str) -> None:
        self.name = name

    def update_child_independent_attrs(self, reference_object: Self) -> None:
        self.name = reference_object.name


class FakeSubclass(FakePylogicObject): ...


class FakeSubclass2(FakePylogicObject):
    child_independent_attrs = FakePylogicObject.child_independent_attrs + (
        "extra_attr",
    )
    __slots__ = FakePylogicObject.child_dependent_attrs + child_independent_attrs


def setup():
    obj1 = FakePylogicObject("1")
    obj2 = FakePylogicObject("2")
    obj3 = FakePylogicObject("3", children=[obj1, obj2])
    obj4 = FakePylogicObject("4", children=[obj3, obj2])
    obj5 = FakePylogicObject("5", children=[obj4, obj1, obj3])
    return obj1, obj2, obj3, obj4, obj5


class TestInitialization:
    def t_init(self):
        FakePylogicObject("1")

    def s_copy(self):
        obj1 = FakePylogicObject("1")

    def t_copy(self):
        copy.copy(obj1)

    s_deepcopy = s_copy

    def t_deepcopy(self):
        copy.deepcopy(obj1)


class TestEquality:
    def s_equal(self):
        _, _, _, _, obj5 = setup()
        _, _, _, _, obj5_copy = setup()

    def t_equal(self):
        obj5 == obj5_copy

    def s_not_equal_different_name(self):
        _, _, _, _, obj5 = setup()
        obj6 = FakePylogicObject("6", children=obj5.children)

    def t_not_equal_different_name(self):
        obj5 != obj6

    def s_not_equal_different_class(self):
        _, _, _, _, obj5 = setup()
        obj5subclass = FakeSubclass("5", children=obj5.children)

    def t_not_equal_different_class(self):
        obj5 != obj5subclass

    s_equal_up_to_subclass = s_not_equal_different_class

    def t_equal_up_to_subclass(self):
        obj5.equal_up_to_subclass(obj5subclass)

    def s_equal_child_independent_attrs(self):
        _, _, _, _, obj5 = setup()
        obj5subclass = FakeSubclass("5", children=[])

    def t_equal_child_independent_attrs(self):
        obj5.eq_child_independent_attrs(obj5subclass)

    def s_hash(self):
        _, _, _, _, obj5 = setup()

    def t_hash(self):
        hash(obj5)


class TestReplace:
    def s_replace_all_positions_none(self):
        obj1, obj2, obj3, obj4, obj5 = setup()

    def t_replace_all_positions_none(self):
        obj5.replace({obj1: obj2})

    s_replace_all_multiple_replacements_positions_none = s_replace_all_positions_none

    def t_replace_all_multiple_replacements_positions_none(self):
        new_obj = obj5.replace({obj1: obj2, obj2: obj3})

    s_replace_all_cyclic_replacements_positions_none = s_replace_all_positions_none

    def t_replace_all_cyclic_replacements_positions_none(self):
        new_obj = obj5.replace({obj1: obj2, obj2: obj1})

    def t_replace_positions_single_empty_root_not_key(self):
        """
        Should replace only the root object if it matches (not in this case).
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj1: obj2}, positions=[[]])
        assert new_obj == FakePylogicObject(
            "5",
            children=[
                FakePylogicObject(
                    "4",
                    children=[
                        FakePylogicObject(
                            "3",
                            children=[FakePylogicObject("1"), FakePylogicObject("2")],
                        ),
                        FakePylogicObject("2"),
                    ],
                ),
                FakePylogicObject("1"),
                FakePylogicObject(
                    "3", children=[FakePylogicObject("1"), FakePylogicObject("2")]
                ),
            ],
        )
        assert obj5.children == [obj4, obj1, obj3]
        assert new_obj == obj5

    def t_replace_positions_single_empty_root_in_keys(self):
        """
        Should replace only the root object if it matches.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj5: obj2}, positions=[[]])
        assert new_obj == FakePylogicObject("2")
        assert obj5.children == [obj4, obj1, obj3]

    def t_replace_positions_has_empty_root_not_key(self):
        """
        Should replace according to the dict and positions.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj1: obj2}, positions=[[], [0]])
        assert new_obj == FakePylogicObject(
            "5",
            children=[
                FakePylogicObject(
                    "4",
                    children=[
                        FakePylogicObject(
                            "3",
                            children=[FakePylogicObject("2"), FakePylogicObject("2")],
                        ),
                        FakePylogicObject("2"),
                    ],
                ),
                FakePylogicObject("1"),
                FakePylogicObject(
                    "3", children=[FakePylogicObject("1"), FakePylogicObject("2")]
                ),
            ],
        )
        assert obj5.children == [obj4, obj1, obj3]

    def t_replace_positions_has_empty_root_in_keys(self):
        """
        Should replace according to the dict and positions.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj5: obj2, obj1: obj2}, positions=[[0], []])
        assert new_obj == FakePylogicObject("2")
        assert obj5.children == [obj4, obj1, obj3]

    def t_replace_positions_empty(self):
        """
        Should not replace anything.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj1: obj2}, positions=[])
        assert new_obj == FakePylogicObject(
            "5",
            children=[
                FakePylogicObject(
                    "4",
                    children=[
                        FakePylogicObject(
                            "3",
                            children=[FakePylogicObject("1"), FakePylogicObject("2")],
                        ),
                        FakePylogicObject("2"),
                    ],
                ),
                FakePylogicObject("1"),
                FakePylogicObject(
                    "3", children=[FakePylogicObject("1"), FakePylogicObject("2")]
                ),
            ],
        )
        assert obj5.children == [obj4, obj1, obj3]
        assert new_obj == obj5

    def t_replace_positions_empty_root_in_keys(self):
        """
        Should not replace anything.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj5: obj2}, positions=[])
        assert new_obj == FakePylogicObject(
            "5",
            children=[
                FakePylogicObject(
                    "4",
                    children=[
                        FakePylogicObject(
                            "3",
                            children=[FakePylogicObject("1"), FakePylogicObject("2")],
                        ),
                        FakePylogicObject("2"),
                    ],
                ),
                FakePylogicObject("1"),
                FakePylogicObject(
                    "3", children=[FakePylogicObject("1"), FakePylogicObject("2")]
                ),
            ],
        )
        assert obj5.children == [obj4, obj1, obj3]
        assert new_obj == obj5

    def t_replace_all_positions_empty_dict(self):
        """
        Should not replace anything.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({})
        assert new_obj == FakePylogicObject(
            "5",
            children=[
                FakePylogicObject(
                    "4",
                    children=[
                        FakePylogicObject(
                            "3",
                            children=[FakePylogicObject("1"), FakePylogicObject("2")],
                        ),
                        FakePylogicObject("2"),
                    ],
                ),
                FakePylogicObject("1"),
                FakePylogicObject(
                    "3", children=[FakePylogicObject("1"), FakePylogicObject("2")]
                ),
            ],
        )
        assert obj5.children == [obj4, obj1, obj3]
        assert new_obj == obj5

    def t_replace_specific_position(self):
        """
        Should replace obj1 with obj2 in the first child of obj5.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj1: obj2}, positions=[[0]])
        assert new_obj == FakePylogicObject(
            "5",
            children=[
                FakePylogicObject(
                    "4",
                    children=[
                        FakePylogicObject(
                            "3",
                            children=[FakePylogicObject("2"), FakePylogicObject("2")],
                        ),
                        FakePylogicObject("2"),
                    ],
                ),
                FakePylogicObject("1"),
                FakePylogicObject(
                    "3", children=[FakePylogicObject("1"), FakePylogicObject("2")]
                ),
            ],
        )
        assert obj5.children == [obj4, obj1, obj3]

    def t_replace_specific_multiple_positions(self):
        """
        Should replace obj1 with obj2 in the first child of the first child of obj5 and
        the third child of obj5.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj1: obj2}, positions=[[0, 0], [2]])
        assert new_obj == FakePylogicObject(
            "5",
            children=[
                FakePylogicObject(
                    "4",
                    children=[
                        FakePylogicObject(
                            "3",
                            children=[FakePylogicObject("2"), FakePylogicObject("2")],
                        ),
                        FakePylogicObject("2"),
                    ],
                ),
                FakePylogicObject("1"),
                FakePylogicObject(
                    "3", children=[FakePylogicObject("2"), FakePylogicObject("2")]
                ),
            ],
        )
        assert obj5.children == [obj4, obj1, obj3]

    def t_replace_specific_position_self_reference(self):
        """
        Should replace obj1 with obj5 at the specific position.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj1: obj5}, positions=[[0]])
        assert new_obj == FakePylogicObject(
            "5",
            children=[
                FakePylogicObject(
                    "4",
                    children=[
                        FakePylogicObject("3", children=[obj5, FakePylogicObject("2")]),
                        FakePylogicObject("2"),
                    ],
                ),
                FakePylogicObject("1"),
                FakePylogicObject(
                    "3", children=[FakePylogicObject("1"), FakePylogicObject("2")]
                ),
            ],
        )
        assert new_obj.children[0].children[0].children[0] is obj5
        assert new_obj.children[0].children[0].children[0] is not new_obj
        assert obj5.children == [obj4, obj1, obj3]

    def t_replace_positions_oob(self):
        """
        Should not replace anything because the specified position is out of bounds.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj1: obj2}, positions=[[0, 4]])
        # child at index 0 does not have a child at index 4, so this
        # should not replace anything
        assert new_obj == FakePylogicObject(
            "5",
            children=[
                FakePylogicObject(
                    "4",
                    children=[
                        FakePylogicObject(
                            "3",
                            children=[FakePylogicObject("1"), FakePylogicObject("2")],
                        ),
                        FakePylogicObject("2"),
                    ],
                ),
                FakePylogicObject("1"),
                FakePylogicObject(
                    "3", children=[FakePylogicObject("1"), FakePylogicObject("2")]
                ),
            ],
        )
        assert obj5.children == [obj4, obj1, obj3]

    def t_replace_user_invalid_internal_args(self):
        """
        Should raise an error when invalid internal arguments are passed.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        try:
            obj5.replace({obj1: obj2}, _path=[], _depth=1)
        except ValueError as e:
            pass
            # assert str(e) == "Depth must be at most the length of the path if path is provided."

    def t_replace_self(self):
        """
        Should replace obj1 with itself in obj5, effectively making no change.
        """
        obj1, _, _, _, obj5 = setup()
        new_obj = obj5.replace({obj1: obj1})
        assert new_obj == obj5

    def t_replace_self_with_copy(self):
        """
        Should replace obj1 with a copy of itself in obj5, effectively making no change.
        """
        obj1, _, _, _, obj5 = setup()
        new_obj = obj5.replace({obj1: copy.copy(obj1)})
        assert new_obj == obj5

    def t_replace_self_with_deepcopy(self):
        """
        Should replace obj1 with a deep copy of itself in obj5, effectively making no
        change.
        """
        obj1, _, _, _, obj5 = setup()
        new_obj = obj5.replace({obj1: copy.deepcopy(obj1)})
        assert new_obj == obj5

    def t_replace_self_self(self):
        """
        Should replace obj5 with itself in obj5, effectively making no change.
        """
        _, _, _, _, obj5 = setup()
        new_obj = obj5.replace({obj5: obj5})
        assert new_obj == obj5
        assert new_obj is obj5

    def t_replace_idempotent_when_key_is_not_child(self):
        """
        When a key in `replace_dict` is not a descendant of its value,
        `replace` should be idempotent.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj1: obj2})
        assert new_obj == new_obj.replace({obj1: obj2})

    def t_replace_not_idempotent_when_key_is_child(self):
        """
        When a key in `replace_dict` is a descendant of its value,
        `replace` should not be idempotent.
        """
        obj1, obj2, obj3, obj4, obj5 = setup()
        new_obj = obj5.replace({obj1: obj5})
        assert new_obj != new_obj.replace({obj1: obj5})

    def t_replace_with_different_equal_check(self):
        """
        Should replace obj1b with obj2 in obj5, but only if the objects are equal
        according to the provided equal_check function.
        """
        obj1, obj2, obj3, obj4, _ = setup()
        obj1b = FakePylogicObject("1")
        obj5 = FakePylogicObject("5", children=[obj4, obj1b, obj3])
        new_obj = obj5.replace({obj1b: obj2}, equal_check=lambda x, y: x is y)
        assert new_obj == FakePylogicObject(
            "5",
            children=[
                FakePylogicObject(
                    "4",
                    children=[
                        FakePylogicObject(
                            "3",
                            children=[FakePylogicObject("1"), FakePylogicObject("2")],
                        ),
                        FakePylogicObject("2"),
                    ],
                ),
                FakePylogicObject("2"),
                FakePylogicObject(
                    "3", children=[FakePylogicObject("1"), FakePylogicObject("2")]
                ),
            ],
        )
        assert obj5.children == [obj4, obj1b, obj3]


class TestDictConstruction:
    def t_to_dict(self):
        """
        Should convert the object to a dictionary representation.
        """
        _, _, _, _, obj5 = setup()
        expected_dict = {
            "children": [
                {
                    "children": [
                        {
                            "children": [
                                {
                                    "children": [],
                                    "leaves": [],
                                    "name": "1",
                                    "class_module": "pylogic_performance_tests.base",
                                    "class_name": "FakePylogicObject",
                                },
                                {
                                    "children": [],
                                    "leaves": [],
                                    "name": "2",
                                    "class_module": "pylogic_performance_tests.base",
                                    "class_name": "FakePylogicObject",
                                },
                            ],
                            "leaves": [
                                {
                                    "children": [],
                                    "leaves": [],
                                    "name": "1",
                                    "class_module": "pylogic_performance_tests.base",
                                    "class_name": "FakePylogicObject",
                                },
                                {
                                    "children": [],
                                    "leaves": [],
                                    "name": "2",
                                    "class_module": "pylogic_performance_tests.base",
                                    "class_name": "FakePylogicObject",
                                },
                            ],
                            "name": "3",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                        {
                            "children": [],
                            "leaves": [],
                            "name": "2",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                    ],
                    "leaves": [
                        {
                            "children": [],
                            "leaves": [],
                            "name": "1",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                        {
                            "children": [],
                            "leaves": [],
                            "name": "2",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                        {
                            "children": [],
                            "leaves": [],
                            "name": "2",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                    ],
                    "name": "4",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
                {
                    "children": [],
                    "leaves": [],
                    "name": "1",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
                {
                    "children": [
                        {
                            "children": [],
                            "leaves": [],
                            "name": "1",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                        {
                            "children": [],
                            "leaves": [],
                            "name": "2",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                    ],
                    "leaves": [
                        {
                            "children": [],
                            "leaves": [],
                            "name": "1",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                        {
                            "children": [],
                            "leaves": [],
                            "name": "2",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                    ],
                    "name": "3",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
            ],
            "leaves": [
                {
                    "children": [],
                    "leaves": [],
                    "name": "1",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
                {
                    "children": [],
                    "leaves": [],
                    "name": "2",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
                {
                    "children": [],
                    "leaves": [],
                    "name": "2",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
                {
                    "children": [],
                    "leaves": [],
                    "name": "1",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
                {
                    "children": [],
                    "leaves": [],
                    "name": "1",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
                {
                    "children": [],
                    "leaves": [],
                    "name": "2",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
            ],
            "name": "5",
            "class_module": "pylogic_performance_tests.base",
            "class_name": "FakePylogicObject",
        }
        assert obj5.to_dict() == expected_dict

    def t_from_dict(self):
        """
        Should create an object from a dictionary representation.
        """
        _, _, _, _, obj5 = setup()
        dict_ = {
            "children": [
                {
                    "children": [
                        {
                            "children": [
                                {
                                    "children": [],
                                    "name": "1",
                                    "class_module": "pylogic_performance_tests.base",
                                    "class_name": "FakePylogicObject",
                                },
                                {
                                    "children": [],
                                    "name": "2",
                                    "class_module": "pylogic_performance_tests.base",
                                    "class_name": "FakePylogicObject",
                                },
                            ],
                            "name": "3",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                        {
                            "children": [],
                            "name": "2",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                    ],
                    "name": "4",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
                {
                    "children": [],
                    "name": "1",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
                {
                    "children": [
                        {
                            "children": [],
                            "name": "1",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                        {
                            "children": [],
                            "name": "2",
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                        },
                    ],
                    "name": "3",
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                },
            ],
            "name": "5",
            "class_module": "pylogic_performance_tests.base",
            "class_name": "FakePylogicObject",
        }
        new_obj = FakePylogicObject.from_dict(dict_)
        assert new_obj == obj5

    def t_dict_to_constructor_kwargs(self):
        """
        Should convert a dictionary representation to constructor arguments.
        """
        _, _, _, _, obj5 = setup()
        dict_ = {
            "class_module": "pylogic_performance_tests.base",
            "class_name": "FakePylogicObject",
            "name": "5",
            "children": [
                {
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                    "name": "4",
                    "children": [
                        {
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                            "name": "3",
                            "children": [
                                {
                                    "class_module": "pylogic_performance_tests.base",
                                    "class_name": "FakePylogicObject",
                                    "name": "1",
                                    "children": [],
                                },
                                {
                                    "class_module": "pylogic_performance_tests.base",
                                    "class_name": "FakePylogicObject",
                                    "name": "2",
                                    "children": [],
                                },
                            ],
                        },
                        {
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                            "name": "2",
                            "children": [],
                        },
                    ],
                },
                {
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                    "name": "1",
                    "children": [],
                },
                {
                    "class_module": "pylogic_performance_tests.base",
                    "class_name": "FakePylogicObject",
                    "name": "3",
                    "children": [
                        {
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                            "name": "1",
                            "children": [],
                        },
                        {
                            "class_module": "pylogic_performance_tests.base",
                            "class_name": "FakePylogicObject",
                            "name": "2",
                            "children": [],
                        },
                    ],
                },
            ],
        }
        kwargs = FakePylogicObject.dict_to_constructor_kwargs(dict_)
        assert kwargs == {
            "name": "5",
            "children": [
                FakePylogicObject(
                    "4",
                    children=[
                        FakePylogicObject(
                            "3",
                            children=[FakePylogicObject("1"), FakePylogicObject("2")],
                        ),
                        FakePylogicObject("2"),
                    ],
                ),
                FakePylogicObject("1"),
                FakePylogicObject(
                    "3", children=[FakePylogicObject("1"), FakePylogicObject("2")]
                ),
            ],
        }


class TestUnify:
    """
    Basic object = object that can serve as a key in the unification dict.
    By default, basic objects are objects with no children. Can be modified with
    the `key_check` parameter.

    Complex object = not a basic object.
    """

    def t_eq_basic(self):
        """
        Should unify two objects that are equal.
        """
        obj1 = FakePylogicObject("1")
        obj1b = FakePylogicObject("1")
        unified = obj1.unify(obj1b)
        assert bool(unified)
        assert unified == {}

    def t_different_children_root_basic(self):
        """
        Should unify two objects.
        """
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3")
        obj1 = FakePylogicObject("1", children=[obj2])
        obj1b = FakePylogicObject("1", children=[obj3])
        unified = obj1.unify(obj1b, key_check=lambda o: o.name == "1")  # type: ignore
        assert unified == {obj1: obj1b}

    def t_eq_complex_leaves_basic(self):
        """
        Should unify two objects that are equal.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2", children=[obj1])
        obj2b = FakePylogicObject("2", children=[obj1])
        unified = obj2.unify(obj2b)
        assert bool(unified)
        assert unified == {}

    def t_eq_complex_no_basic(self):
        """
        Should unify two objects that are equal.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2", children=[obj1])
        obj2b = FakePylogicObject("2", children=[obj1])
        unified = obj2.unify(obj2b, key_check=lambda _: False)  # type: ignore
        assert bool(unified)
        assert unified == {}

    def t_complex_no_basic(self):
        """
        Should be None when unequal and no basic.
        """
        obj1 = FakePylogicObject("1")
        obj3 = FakePylogicObject("3")
        obj2 = FakePylogicObject("2", children=[obj3])
        obj2b = FakePylogicObject("2", children=[obj1])
        unified = obj2.unify(obj2b, key_check=lambda _: False)  # type: ignore
        assert unified is None

    def t_basic_to_complex(self):
        """
        basic.unify(complex) should result in {basic: complex}
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2", children=[obj1])
        obj3 = FakePylogicObject("3", children=[obj1, obj2])
        obj4 = FakePylogicObject("4")
        unified = obj4.unify(obj3)
        assert unified == {obj4: obj3}

    def t_no_basic_on_lhs(self):
        """
        If `unify` is called on something that has no basic objects, should
        result in `None`.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2", children=[obj1])
        obj3 = FakePylogicObject("3", children=[obj1, obj2])
        obj4 = FakePylogicObject("4")
        unified = obj4.unify(obj3, key_check=lambda o: o.name == "1")  # type: ignore
        assert unified is None

    def t_complex_unification(self):
        """
        Should find a unification, if possible.
        """
        # obj1 unifies to obj1b but should not be included in the unification dict
        # because they are equal.
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3")
        obj4 = FakePylogicObject("4", children=[obj2, obj3])
        obj5 = FakePylogicObject("5", children=[obj4, obj1])

        obj1b = FakePylogicObject("1")
        objd = FakePylogicObject("d")
        obje = FakePylogicObject("e")
        objf = FakePylogicObject("f")
        objg = FakePylogicObject("g")
        objh = FakePylogicObject("h")
        objb = FakePylogicObject("b", children=[objd, obje, objf])
        objc = FakePylogicObject("c", children=[objg, objh])
        obj4b = FakePylogicObject("4", children=[objb, objc])
        obj5b = FakePylogicObject("5", children=[obj4b, obj1b])

        unified = obj5.unify(obj5b)
        assert unified == {obj2: objb, obj3: objc}

    def t_complex_unification_different_keycheck(self):
        """
        Should find a unification, if possible, with a different `key_check`
        parameter.
        """
        var_obj1 = FakePylogicObject("var_1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3")
        var_obj4 = FakePylogicObject("var_4", children=[obj2, obj3])
        obj5 = FakePylogicObject("5", children=[var_obj4, var_obj1])

        obj1b = FakePylogicObject("1")
        objd = FakePylogicObject("d")
        obje = FakePylogicObject("e")
        objf = FakePylogicObject("f")
        objg = FakePylogicObject("g")
        objh = FakePylogicObject("h")
        objb = FakePylogicObject("b", children=[objd, obje, objf])
        objc = FakePylogicObject("c", children=[objg, objh])
        obj4b = FakePylogicObject("4", children=[objb, objc])
        obj5b = FakePylogicObject("5", children=[obj4b, obj1b])

        unified = obj5.unify(obj5b, key_check=lambda o: o.name.startswith("var_"))  # type: ignore
        assert unified == {var_obj1: obj1b, var_obj4: obj4b}

    def t_complex_unification_impossible(self):
        """
        Should find a unification, if possible (not in this case).
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3")
        obj4 = FakePylogicObject("4", children=[obj2, obj3])
        obj5 = FakePylogicObject("5", children=[obj4, obj1])

        obj1b = FakePylogicObject("1")
        objd = FakePylogicObject("d")
        obje = FakePylogicObject("e")
        objf = FakePylogicObject("f")
        objg = FakePylogicObject("g")
        objh = FakePylogicObject("h")
        obji = FakePylogicObject("i")  # extra child
        objb = FakePylogicObject("b", children=[objd, obje, objf])
        objc = FakePylogicObject("c", children=[objg, objh])
        obj4b = FakePylogicObject("4", children=[objb, objc, obji])
        obj5b = FakePylogicObject("5", children=[obj4b, obj1b])

        unified = obj5.unify(obj5b)
        assert unified is None

    def t_unification_replacement(self):
        """
        Check that unification followed by replacement works as expected.
        ```
        unif = a.unify(b)
        a.replace(unif) == b
        ```
        """
        # obj1 unifies to obj1b but should not be included in the unification dict
        # because they are equal.
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3")
        obj4 = FakePylogicObject("4", children=[obj2, obj3])
        obj5 = FakePylogicObject("5", children=[obj4, obj1])

        obj1b = FakePylogicObject("1")
        objd = FakePylogicObject("d")
        obje = FakePylogicObject("e")
        objf = FakePylogicObject("f")
        objg = FakePylogicObject("g")
        objh = FakePylogicObject("h")
        objb = FakePylogicObject("b", children=[objd, obje, objf])
        objc = FakePylogicObject("c", children=[objg, objh])
        obj4b = FakePylogicObject("4", children=[objb, objc])
        obj5b = FakePylogicObject("5", children=[obj4b, obj1b])

        unified = obj5.unify(obj5b)
        assert unified == {obj2: objb, obj3: objc}
        assert obj5.replace(unified) == obj5b  # type: ignore


class TestMultiUnify:
    def t_multi_unify_on_multivar(self):
        """
        When called on a multi-var (key_for_list_check is always True),
        should result in a list.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2")

        unified = obj1.multi_unify(
            obj2, key_for_list_check=lambda o: o.name == "1"  # type: ignore
        )
        assert unified == {obj1: [obj2]}

    def t_multivar_is_empty(self):
        """
        Should unify a multivar with an empty list when appropriate.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3")
        obj4 = FakePylogicObject("4", children=[obj1, obj2, obj3])

        obj1b = FakePylogicObject("1b")
        obj3b = FakePylogicObject("3b")
        obj4b = FakePylogicObject("4", children=[obj1b, obj3b])  # name should match

        unified = obj4.multi_unify(
            obj4b, key_for_list_check=lambda o: o.name == "2"  # type: ignore
        )
        assert unified == {obj1: obj1b, obj2: [], obj3: obj3b}

    def t_complex(self):
        r"""
        Should multi-unify two complex objects. These cannot be unified the regular
        way. Unify
        ```
                1
               / \
              2   c
             /
            d
        with
                     1
                   / | \
                  2  3  4
                 /\     |
                5  6    7
        ```
        where c and d can match a list of objects, to get
        {c: [3, 4], d: [5, 6]}
        """
        d = FakePylogicObject("d")
        obj2_pattern = FakePylogicObject("2", children=[d])
        c = FakePylogicObject("c")
        obj1_pattern = FakePylogicObject("1", children=[obj2_pattern, c])

        obj5 = FakePylogicObject("5")
        obj6 = FakePylogicObject("6")
        obj7 = FakePylogicObject("7")
        obj2 = FakePylogicObject("2", children=[obj5, obj6])
        obj3 = FakePylogicObject("3")
        obj4 = FakePylogicObject("4", children=[obj7])
        obj1 = FakePylogicObject("1", children=[obj2, obj3, obj4])

        unified = obj1_pattern.multi_unify(
            obj1, key_for_list_check=lambda o: o.name in ["c", "d"]  # type: ignore
        )
        assert unified == {c: [obj3, obj4], d: [obj5, obj6]}

    def t_complex_unification_different_keycheck(self):
        """
        Multi-unify two complex objects with a different `key_check` parameter.
        Should be same as unify with a different `key_check` parameter.
        """
        var_obj1 = FakePylogicObject("var_1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3")
        var_obj4 = FakePylogicObject("var_4", children=[obj2, obj3])
        obj5 = FakePylogicObject("5", children=[var_obj4, var_obj1])

        obj1b = FakePylogicObject("1")
        objd = FakePylogicObject("d")
        obje = FakePylogicObject("e")
        objf = FakePylogicObject("f")
        objg = FakePylogicObject("g")
        objh = FakePylogicObject("h")
        objb = FakePylogicObject("b", children=[objd, obje, objf])
        objc = FakePylogicObject("c", children=[objg, objh])
        obj4b = FakePylogicObject("4", children=[objb, objc])
        obj5b = FakePylogicObject("5", children=[obj4b, obj1b])

        unified = obj5.multi_unify(obj5b, key_check=lambda o: o.name.startswith("var_"))  # type: ignore
        assert unified == {var_obj1: obj1b, var_obj4: obj4b}


class TestStringMatch:
    def t_all_multi_var(self):
        pattern = ["*a", "*b"]
        target = [1, 2, 3]
        is_multi_var = lambda x: x.startswith("*")
        expected = [
            {"*a": (0, 0), "*b": (0, 3)},
            {"*a": (0, 1), "*b": (1, 3)},
            {"*a": (0, 2), "*b": (2, 3)},
            {"*a": (0, 3), "*b": (3, 3)},
        ]
        actual = string_match(pattern, target, is_multi_var=is_multi_var)
        assert actual == expected

        expected = [
            {"*a": [], "*b": [1, 2, 3]},
            {"*a": [1], "*b": [2, 3]},
            {"*a": [1, 2], "*b": [3]},
            {"*a": [1, 2, 3], "*b": []},
        ]
        actual = matches_to_actual(actual, target)
        assert actual == expected

    def t_typical(self):
        pattern = ["a", "*b", "c", "*d"]
        target = [1, 2, 3]
        is_multi_var = lambda x: x.startswith("*")
        expected = [
            {"a": 0, "*b": (1, 1), "c": 1, "*d": (2, 3)},
            {"a": 0, "*b": (1, 2), "c": 2, "*d": (3, 3)},
        ]
        actual = string_match(pattern, target, is_multi_var=is_multi_var)
        assert actual == expected

        expected = [
            {"a": 1, "*b": [], "c": 2, "*d": [3]},
            {"a": 1, "*b": [2], "c": 3, "*d": []},
        ]
        actual = matches_to_actual(actual, target)
        assert actual == expected

    def t_no_match(self):
        pattern = ["a", "c"]
        target = [1, 2, 3]
        is_multi_var = lambda x: x.startswith("*")
        expected = []
        actual = string_match(pattern, target, is_multi_var=is_multi_var)
        assert actual == expected

    def t_matches_to_actual_empty(self):
        expected = []
        actual = matches_to_actual([], [1, 2, 3])
        assert actual == expected


class TestSubobjectFind:
    def t_root(self):
        """
        Should find the root object in the object tree.
        """
        obj = FakePylogicObject("1")
        path = obj.subobject_find(obj)
        assert path == []

    def t_first_child(self):
        """
        Should find the first child in the object tree.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2", children=[obj1])
        path = obj2.subobject_find(obj1)
        assert path == [0]

    def t_typical(self):
        """
        Should find a subobject in the object tree.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3", children=[obj1])
        obj4 = FakePylogicObject("4", children=[obj2, obj3])
        path = obj4.subobject_find(obj1)
        assert path == [1, 0]

    def t_typical_non_leaf(self):
        """
        Should find a non-leaf subobject in the object tree.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3", children=[obj1])
        obj4 = FakePylogicObject("4", children=[obj2, obj3])
        obj5 = FakePylogicObject("5", children=[obj1, obj4])
        obj6 = FakePylogicObject("6", children=[obj5])
        path = obj6.subobject_find(obj3)
        assert path == [0, 1, 1]

    def t_non_existent(self):
        """
        Should result in None if the subobject does not exist in the object tree.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3", children=[obj1])
        obj4 = FakePylogicObject("4", children=[obj2, obj3])
        path = obj4.subobject_find(FakePylogicObject("5"))
        assert path is None


class TestSubobjectFindAll:
    def t_root(self):
        """
        Should find the root object in the object tree.
        """
        obj = FakePylogicObject("1")
        paths = obj.subobject_find_all(obj)
        assert paths == [[]]

    def t_first_child(self):
        """
        Should find the first child in the object tree.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2", children=[obj1])
        paths = obj2.subobject_find_all(obj1)
        assert paths == [[0]]

    def t_typical(self):
        """
        Should find a subobject in the object tree.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3", children=[obj1])
        obj4 = FakePylogicObject("4", children=[obj2, obj1])
        obj5 = FakePylogicObject("5", children=[obj1])
        obj6 = FakePylogicObject("6", children=[obj2, obj3, obj4, obj5])
        paths = obj6.subobject_find_all(obj1)
        assert paths == [[1, 0], [2, 1], [3, 0]]

    def t_typical_non_leaf(self):
        """
        Should find a non-leaf subobject in the object tree.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3", children=[obj1])
        obj4 = FakePylogicObject("4", children=[obj2, obj3])
        obj5 = FakePylogicObject("5", children=[obj1, obj4])
        obj6 = FakePylogicObject("6", children=[obj5, obj3])
        paths = obj6.subobject_find_all(obj3)
        assert paths == [[0, 1, 1], [1]]

    def t_non_existent(self):
        """
        Should result in an empty list if the subobject does not exist in the object tree.
        """
        obj1 = FakePylogicObject("1")
        obj2 = FakePylogicObject("2")
        obj3 = FakePylogicObject("3", children=[obj1])
        obj4 = FakePylogicObject("4", children=[obj2, obj3])
        paths = obj4.subobject_find_all(FakePylogicObject("5"))
        assert paths == []
