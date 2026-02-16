import math
import operator
from collections import deque
from typing import Self, Optional, Callable, TYPE_CHECKING

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
    """Algebraic expressions as specified by the Kappa language.

    Attributes:
        type: Type of expression (literal, variable, binary_op, etc.).
        attrs: Dictionary of attributes specific to the expression type.
    """

    @classmethod
    def from_kappa(cls, kappa_str: str) -> Self:
        """Parse an Expression from a Kappa string.

        Raises:
            AssertionError: If the string doesn't represent a valid expression.
        """
        from pykappa.grammar import kappa_parser, parse_tree_to_expression

        input_tree = kappa_parser.parse(kappa_str)
        assert input_tree.data == "kappa_input"
        expr_tree = input_tree.children[0]
        assert expr_tree.data in ["!algebraic_expression", "algebraic_expression"]
        return parse_tree_to_expression(expr_tree)

    def __init__(self, type, **attrs):
        self.type = type
        self.attrs = attrs

    @property
    def kappa_str(self) -> str:
        """Get the expression representation in Kappa format.

        Raises:
            ValueError: If expression type is not supported for string conversion.
        """
        if self.type == "literal":
            return str(self.evaluate())

        elif self.type == "boolean_literal":
            return "[true]" if self.attrs["value"] else "[false]"

        elif self.type == "variable":
            return f"'{self.attrs["name"]}'"

        elif self.type in ("binary_op", "comparison"):
            left_str = self.attrs["left"].kappa_str
            right_str = self.attrs["right"].kappa_str
            return f"({left_str}) {self.attrs['operator']} ({right_str})"

        elif self.type == "unary_op":
            return f"{self.attrs['operator']} ({self.attrs['child'].kappa_str})"

        elif self.type == "list_op":
            children_str = " ".join(
                f"({child.kappa_str})" for child in self.attrs["children"]
            )
            return f"{self.attrs["operator"]} {children_str}"

        elif self.type == "defined_constant":
            return f"{self.attrs["name"]}"

        elif self.type == "parentheses":
            return self.attrs["child"].kappa_str

        elif self.type == "conditional":
            true_expr_str = self.attrs["true_expr"].kappa_str
            false_expr_str = self.attrs["false_expr"].kappa_str
            return f"{self.attrs["condition"].kappa_str} [?] {true_expr_str} [:] {false_expr_str}"

        elif self.type in ("logical_or", "logical_and"):
            left_str = self.attrs["left"].kappa_str
            right_str = self.attrs["right"].kappa_str
            op = {"logical_or": "||", "logical_and": "&&"}
            return f"({left_str}) {op[self.type]} ({right_str})"

        elif self.type == "logical_not":
            return f"[not] ({self.attrs['child'].kappa_str})"

        elif self.type == "reserved_variable":
            return self.attrs["value"].kappa_str

        elif self.type == "component_pattern":
            return f"|{self.attrs['value'].kappa_str}|"

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

        elif self.type == "variable":
            name = self.attrs["name"]
            if system is None:
                raise ValueError(f"{self} needs a System to evaluate variable '{name}'")
            return system[name]

        elif self.type in ("binary_op", "comparison"):
            left_val = self.attrs["left"].evaluate(system)
            right_val = self.attrs["right"].evaluate(system)
            return parse_operator(self.attrs["operator"])(left_val, right_val)

        elif self.type == "unary_op":
            child_val = self.attrs["child"].evaluate(system)
            return parse_operator(self.attrs["operator"])(child_val)

        elif self.type == "list_op":
            children_vals = [child.evaluate(system) for child in self.attrs["children"]]
            return parse_operator(self.attrs["operator"])(children_vals)

        elif self.type == "defined_constant":
            const = self.attrs["name"]
            if const == "[pi]":
                return math.pi
            else:
                raise ValueError(f"Unknown constant: {const}")

        elif self.type == "parentheses":
            return self.attrs["child"].evaluate(system)

        elif self.type == "conditional":
            cond_val = self.attrs["condition"].evaluate(system)
            return (
                self.attrs["true_expr"].evaluate(system)
                if cond_val
                else self.attrs["false_expr"].evaluate(system)
            )

        elif self.type == "logical_or":
            left_val = self.attrs["left"].evaluate(system)
            right_val = self.attrs["right"].evaluate(system)
            return left_val or right_val

        elif self.type == "logical_and":
            left_val = self.attrs["left"].evaluate(system)
            right_val = self.attrs["right"].evaluate(system)
            return left_val and right_val

        elif self.type == "logical_not":
            return not self.attrs["child"].evaluate(system)

        elif self.type == "reserved_variable":
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
            else:
                raise NotImplementedError(
                    f"Reserved variable {value.type} not implemented yet."
                )

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
                    if isinstance(attr_value, Expression):
                        stack.append(attr_value)
                    elif isinstance(attr_value, (list, tuple)):
                        stack.extend(v for v in attr_value if isinstance(v, Expression))

        return result
