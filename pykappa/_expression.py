import math
import operator
from collections import deque
from typing import Any, Self, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from pykappa.pattern import Component
    from pykappa.system import System


string_to_operator = {
    # Unary
    "[log]": math.log,
    "[exp]": math.exp,
    "[sin]": math.sin,
    "[cos]": math.cos,
    "[tan]": math.tan,
    "[sqrt]": math.sqrt,
    # Binary
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "^": operator.pow,
    "mod": operator.mod,
    # Comparisons
    "=": operator.eq,
    "<": operator.lt,
    ">": operator.gt,
    # List
    "[max]": max,
    "[min]": min,
}


def parse_operator(kappa_operator: str) -> Callable:
    """Convert a Kappa string operator to a Python function.

    Raises:
        ValueError: If the operator is not recognized.
    """
    try:
        return string_to_operator[kappa_operator]
    except KeyError:
        raise ValueError(f"Unknown operator: {kappa_operator}")


class Expression:
    """Algebraic expressions as specified by the Kappa language."""

    type: Any  # Type of expression (literal, variable, binary_op, etc.)
    attrs: dict[str, Any]  # Dictionary of attributes specific to the expression type

    @classmethod
    def from_kappa(cls, kappa_str: str) -> "Expression":
        """Parse an Expression from a Kappa string.

        Raises:
            AssertionError: If the string doesn't represent a valid expression.
        """
        from pykappa._parsing import kappa_parser, ExpressionTransformer

        input_tree = kappa_parser.parse(kappa_str)
        assert input_tree.data == "kappa_input"
        expr_tree = input_tree.children[0]
        assert expr_tree.data in ["!algebraic_expression", "algebraic_expression"]
        return ExpressionTransformer.from_tree(expr_tree)

    def __init__(self, type, **attrs):
        self.type = type
        self.attrs = attrs

    def __str__(self):
        return self.kappa_str

    @property
    def kappa_str(self) -> str:
        """
        Raises:
            ValueError: If expression type is not supported for string conversion.
        """
        if self.type == "literal":
            return str(self.evaluate())
        if self.type == "boolean_literal":
            return "[true]" if self.attrs["value"] else "[false]"
        if self.type == "variable":
            return f"'{self.attrs['name']}'"
        if self.type in ("binary_op", "comparison", "logical_or", "logical_and"):
            operator = {
                "logical_or": "||",
                "logical_and": "&&",
            }.get(self.type, self.attrs.get("operator"))
            return f"({self.attrs['left'].kappa_str}) {operator} ({self.attrs['right'].kappa_str})"
        if self.type in ("unary_op", "logical_not"):
            operator = "[not]" if self.type == "logical_not" else self.attrs["operator"]
            return f"{operator} ({self.attrs['child'].kappa_str})"
        if self.type == "list_op":
            children = " ".join(
                f"({child.kappa_str})" for child in self.attrs["children"]
            )
            return f"{self.attrs['operator']} {children}"
        if self.type == "defined_constant":
            return self.attrs["name"]
        if self.type == "parentheses":
            return self.attrs["child"].kappa_str
        if self.type == "conditional":
            return (
                f"{self.attrs['condition'].kappa_str} [?] {self.attrs['true_expr'].kappa_str} "
                f"[:] {self.attrs['false_expr'].kappa_str}"
            )
        if self.type == "reserved_variable":
            return self.attrs["value"].kappa_str
        if self.type == "component_pattern":
            return f"|{self.attrs['value'].kappa_str}|"
        if self.type == "token_value":
            return f"|{self.attrs['name']}|"

        raise ValueError(f"Unsupported node type: {self.type}")

    def evaluate(self, system: Optional["System"] = None) -> int | float:
        """Evaluate the expression to get its value.

        Args:
            system: System context for variable evaluation (required for variables).

        Raises:
            ValueError: If evaluation fails due to missing context or unsupported type.
        """
        if self.type in ("literal", "boolean_literal"):
            return self.attrs["value"]
        if self.type == "variable":
            name = self.attrs["name"]
            if system is None:
                raise ValueError(f"{self} needs a System to evaluate variable '{name}'")
            return system[name]
        if self.type in ("binary_op", "comparison", "logical_or", "logical_and"):
            left = self.attrs["left"].evaluate(system)
            right = self.attrs["right"].evaluate(system)
            if self.type == "logical_or":
                return left or right
            if self.type == "logical_and":
                return left and right
            return parse_operator(self.attrs["operator"])(left, right)
        if self.type in ("unary_op", "logical_not"):
            child = self.attrs["child"].evaluate(system)
            return (
                not child
                if self.type == "logical_not"
                else parse_operator(self.attrs["operator"])(child)
            )
        if self.type == "list_op":
            children = [child.evaluate(system) for child in self.attrs["children"]]
            return parse_operator(self.attrs["operator"])(children)
        if self.type == "defined_constant":
            const = self.attrs["name"]
            if const == "[pi]":
                return math.pi
            raise ValueError(f"Unknown constant: {const}")
        if self.type == "parentheses":
            return self.attrs["child"].evaluate(system)
        if self.type == "conditional":
            return (
                self.attrs["true_expr"].evaluate(system)
                if self.attrs["condition"].evaluate(system)
                else self.attrs["false_expr"].evaluate(system)
            )
        if self.type == "reserved_variable":
            value = self.attrs["value"]
            if value.type == "component_pattern":
                component: Component = value.attrs["value"]
                if system is None:
                    raise ValueError(
                        f"{self} needs a System to evaluate pattern {component}"
                    )
                return (
                    len(system.mixture.embeddings(component))
                    // component.n_automorphisms
                )
            raise NotImplementedError(
                f"Reserved variable {value.type} not implemented yet."
            )
        if self.type == "token_value":
            name = self.attrs["name"]
            if system is None:
                raise ValueError(f"{self} needs a System to evaluate token '{name}'")
            return system.tokens.get(name, 0.0)

        raise ValueError(f"Unsupported node type: {self.type}")

    def filter(self, type_str: str) -> list[Self]:
        """
        Returns all nodes in the expression tree whose type matches the provided string.

        Note:
            Doesn't detect nodes indirectly nested in named variables.
        """
        result = []
        stack = deque([self])  # DFS from the root

        while stack:
            node = stack.pop()
            if node.type == type_str:
                result.append(node)

            # Add child nodes to the stack
            if hasattr(node, "attrs"):
                for attr_value in node.attrs.values():
                    if isinstance(attr_value, type(self)):
                        stack.append(attr_value)
                    elif isinstance(attr_value, (list, tuple)):
                        stack.extend(v for v in attr_value if isinstance(v, type(self)))

        return result
