from pathlib import Path
from lark import Lark, ParseTree, Tree, Token, Transformer

from pykappa.pattern import Site, Agent, Pattern, SiteType
from pykappa.rule import KappaRule, KappaRuleUnimolecular, KappaRuleBimolecular
from pykappa.expression import Expression


class KappaParser:
    """Parser for Kappa language files and expressions.

    Note:
        Don't instantiate directly: use the global kappa_parser instance.
    """

    def __init__(self):
        """Initialize the Lark parser with Kappa grammar."""
        self._parser = Lark.open(
            str(Path(__file__).parent / "kappa.lark"),
            rel_to=__file__,
            parser="earley",
            lexer="dynamic",
            start="kappa_input",
            propagate_positions=False,
            maybe_placeholders=False,
        )

    def parse(self, text: str) -> ParseTree:
        return self._parser.parse(text)

    def parse_file(self, filepath: str) -> ParseTree:
        with open(filepath, "r") as file:
            return self._parser.parse(file.read())


kappa_parser = KappaParser()


class KappaTransformer(Transformer):
    """Transforms Lark parse trees into Kappa objects (Site, Agent, Pattern, Rule)."""

    # --- Site transformations ---
    def site_name(self, children):
        return str(children[0])

    def agent_name(self, children):
        return str(children[0])

    def state(self, children):
        match children[0]:
            case "#":
                return ("state", "#")
            case str(state):
                return ("state", str(state))
            case Tree(data="unspecified"):
                return ("state", "?")
            case _:
                raise ValueError(f"Unexpected internal state: {children[0]}")

    def partner(self, children):
        match children:
            case ["#"]:
                return ("partner", "#")
            case ["_"]:
                return ("partner", "_")
            case ["."]:
                return ("partner", ".")
            case [Token("INT", x)]:
                return ("partner", int(x))
            case [site_name, agent_name]:
                return ("partner", SiteType(str(site_name), str(agent_name)))
            case [Tree(data="unspecified")]:
                return ("partner", "?")
            case _:
                raise ValueError(f"Unexpected link state: {children}")

    def site(self, children):
        # children[0] is site_name
        # children[1] and children[2] are tuples: ("state"|"partner", value)
        site_name = children[0]
        state = "?"
        partner = "?"

        for child in children[1:]:
            if isinstance(child, tuple):
                tag, value = child
                if tag == "state":
                    state = value
                elif tag == "partner":
                    partner = value

        site = Site(label=site_name, state=state, partner=partner)
        return site

    def undetermined(self, children):
        return Tree("unspecified", [])

    # --- Agent transformations ---
    def interface(self, children):
        return children  # List of sites

    def agent(self, children):
        # First child is agent_name, second is interface (list of sites)
        agent_type = children[0]
        sites = children[1] if len(children) > 1 else []

        agent = Agent(type=agent_type, sites=sites)
        for site in agent:
            site.agent = agent
        return agent

    # --- Pattern transformations ---
    def pattern(self, children):
        agents = [c for c in children if isinstance(c, Agent)]
        return Pattern(agents=agents)

    # --- Rule transformations ---
    def label(self, children):
        return children[0].value.strip("'\"")

    def rate(self, children):
        # Transform the algebraic_expression tree into an Expression
        expr_tree = children[0]
        return ExpressionTransformer().transform(expr_tree)

    def rule_expression(self, children):
        mid_idx = next(
            (i for i, child in enumerate(children) if child in ["->", "<->"])
        )

        left_agents = []
        right_agents = []

        for i, child in enumerate(children):
            if i == mid_idx:
                continue

            if child == ".":
                agent = None
            elif isinstance(child, Agent):
                agent = child
            else:
                continue

            if i < mid_idx:
                left_agents.append(agent)
            else:
                right_agents.append(agent)

        return (Pattern(left_agents), Pattern(right_agents))

    def rev_rule_expression(self, children):
        return self.rule_expression(children)

    def f_rule(self, children):
        # children: [optional label], patterns tuple, [optional token], rate
        patterns = None
        rates = []

        for child in children:
            if isinstance(child, tuple):
                patterns = child
            elif isinstance(child, Expression):
                rates.append(child)

        left, right = patterns
        return [KappaRule(left, right, rates[0])]

    def fr_rule(self, children):
        patterns = None
        rates = []

        for child in children:
            if isinstance(child, tuple):
                patterns = child
            elif isinstance(child, Expression):
                rates.append(child)

        left, right = patterns
        return [KappaRule(left, right, rates[0]), KappaRule(right, left, rates[1])]

    def ambi_rule(self, children):
        patterns = None
        rates = []

        for child in children:
            if isinstance(child, tuple):
                patterns = child
            elif isinstance(child, Expression):
                rates.append(child)

        left, right = patterns
        rules = []

        try:
            if rates[0].evaluate() != 0:
                rules.append(KappaRuleBimolecular(left, right, rates[0]))
        except:
            rules.append(KappaRuleBimolecular(left, right, rates[0]))

        try:
            if rates[1].evaluate() != 0:
                rules.append(KappaRuleUnimolecular(left, right, rates[1]))
        except:
            rules.append(KappaRuleUnimolecular(left, right, rates[1]))

        return rules

    def ambi_fr_rule(self, children):
        patterns = None
        rates = []

        for child in children:
            if isinstance(child, str):
                label = child
            elif isinstance(child, tuple):
                patterns = child
            elif isinstance(child, Expression):
                rates.append(child)

        left, right = patterns
        rules = []

        try:
            if rates[0].evaluate() != 0:
                rules.append(KappaRuleBimolecular(left, right, rates[0]))
        except:
            rules.append(KappaRuleBimolecular(left, right, rates[0]))

        try:
            if rates[1].evaluate() != 0:
                rules.append(KappaRuleUnimolecular(left, right, rates[1]))
        except:
            rules.append(KappaRuleUnimolecular(left, right, rates[1]))

        rules.append(KappaRule(right, left, rates[2]))

        return rules


class ExpressionTransformer(Transformer):
    """Transforms a Lark ParseTree into an Expression object."""

    def algebraic_expression(self, children):
        if len(children) == 1:
            return children[0]
        elif len(children) == 3 and children[0] == "(" and children[2] == ")":
            return children[1]
        else:
            raise Exception(f"Invalid algebraic expression: {children}")

    # --- Literals ---
    def SIGNED_FLOAT(self, token):
        return Expression("literal", value=float(token.value))

    def SIGNED_INT(self, token):
        return Expression("literal", value=int(token.value))

    # --- Variables/Constants ---
    def declared_variable_name(self, children):
        return Expression("variable", name=children[0].value.strip("'\""))

    def reserved_variable_name(self, children):
        return Expression("reserved_variable", value=children[0])

    def pattern(self, children):
        # Use KappaTransformer for pattern parsing
        pattern = KappaTransformer().transform(Tree("pattern", children))
        assert (
            len(pattern.components) == 1
        ), f"The pattern {pattern} must consist of a single component."
        component = pattern.components[0]
        return Expression("component_pattern", value=component)

    def defined_constant(self, children):
        return Expression("defined_constant", name=children[0].value)

    # --- Operations ---
    def binary_op_expression(self, children):
        left, op, right = children
        return Expression("binary_op", operator=op, left=left, right=right)

    def binary_op(self, children):
        return children[0]

    def unary_op_expression(self, children):
        op, child = children
        return Expression("unary_op", operator=op, child=child)

    def unary_op(self, children):
        return children[0]

    def list_op(self, children):
        return children[0].value

    def list_op_expression(self, children):
        op, *args = children
        return Expression("list_op", operator=op, children=args)

    # --- Ternary Conditional ---
    def conditional_expression(self, children):
        cond, true_expr, false_expr = children
        return Expression(
            "conditional", condition=cond, true_expr=true_expr, false_expr=false_expr
        )

    # --- Boolean Logic ---
    def boolean_expression(self, children):
        if len(children) == 3:
            left, op, right = children
            return Expression("comparison", operator=op.value, left=left, right=right)
        # Handle other boolean cases
        return children[0]

    def TRUE(self, token):
        return Expression("boolean_literal", value=True)

    def FALSE(self, token):
        return Expression("boolean_literal", value=False)


def parse_tree_to_expression(tree: Tree) -> Expression:
    return ExpressionTransformer().transform(tree)
