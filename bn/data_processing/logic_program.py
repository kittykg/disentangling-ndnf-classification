from dataclasses import dataclass, InitVar, field
from itertools import product
import random
from pathlib import Path


State = list[int]
Transition = tuple[State, State]


@dataclass(eq=True, frozen=True)
class BodyAtom:
    var_id: int
    val: int
    delay: int

    def to_string(self, var_names: list[str] | None = None):
        if var_names is not None:
            return f"{var_names[self.var_id]}({self.val},T-{self.delay})"
        return f"v_{self.var_id}({self.val},T-{self.delay})"

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()


@dataclass(eq=True, frozen=True)
class Rule:
    head_var_id: int
    head_val: int
    body: frozenset[BodyAtom] = field(init=False)
    body_init: InitVar[list[BodyAtom]]

    def __post_init__(self, body_init: list[BodyAtom]):
        object.__setattr__(self, "body", frozenset(body_init))

    def to_string(self, var_names: list[str] | None = None):
        if var_names is not None:
            output = f"{var_names[self.head_var_id]}({self.head_val},T) :- "
        else:
            output = f"v_{self.head_var_id}({self.head_val},T) :- "
        for b in self.body:
            output += b.to_string(var_names) + ","
        return output[:-1] + "."

    def matches(self, state: list[int]):
        """
        Check if the conditions of the rules holds in the given state
        """
        for atom in self.body:
            # delayed condition
            if atom.var_id >= len(state):
                return False

            if state[atom.var_id] != atom.val:
                return False
        return True

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()


class LogicProgram:
    """
    Define a logic program, a set of rules over variables/values
    encoding the dynamics of a discrete dynamic system.
    """

    variables: dict[int, list[int]]
    variable_names: list[str]
    rules: list[Rule]

    def __init__(
        self,
        variables: dict[int, list[int]],
        variable_names: list[str],
        rules: list[Rule],
    ):
        self.variables = variables.copy()
        self.variable_names = variable_names.copy()
        self.rules = rules.copy()

    @staticmethod
    def load_from_file(file_path: Path) -> "LogicProgram":
        """
        Loads a logic program from a formatted text file and construct a logic
        program instance from it.
            - Extract the list of variables of the system and their value domain
            - Extract a list of rules describing the system dynamics
        """
        with open(file_path, "r") as f:
            lines = list(filter(lambda x: x != "", f.read().split("\n")))

        # Split the file into two parts: variables and rules
        variables_list = [l for l in lines if l.startswith("VAR")]
        rules_list = [l for l in lines if not l.startswith("VAR")]

        return LogicProgram.load_from_strings(variables_list, rules_list)

    @staticmethod
    def load_from_strings(variables_list: list[str], rules_list: list[str]):
        variables: dict[int, list[int]] = {}
        variable_names: list[str] = []
        rules: list[Rule] = []

        # 1. Extract variables
        for l in variables_list:
            # VAR variable_name value1 value2 ...
            tokens = l.split()
            variable_names.append(tokens[1])
            var_id = len(variable_names) - 1
            variables[var_id] = [int(tokens[i]) for i in range(2, len(tokens))]

        # 2.Extract rules
        for l in rules_list:
            # head_var(head_val,T) :- body_var1(body_val1,T-n)...
            tokens = l.split()
            if len(tokens) == 0:
                continue

            # Extract head variable
            head_variable, rest = tokens[0].split("(")
            if head_variable not in variable_names:
                raise ValueError(f"Head variable {head_variable} not declared")
            head_var_id = variable_names.index(head_variable)

            # Extract head value
            head_val, rest = rest.split(",")
            head_val = int(head_val)
            if int(head_val) not in variables[head_var_id]:
                raise ValueError(
                    f"Head value {head_val} not declared for "
                    f"variable {head_variable}"
                )

            # Extract body
            body: list[BodyAtom] = []
            for i in range(2, len(tokens)):
                condition = tokens[i][:-1]

                # Extract variable
                var, rest = condition.split("(")
                if var not in variable_names:
                    raise ValueError(f"Variable {var} not declared")
                var_id = variable_names.index(var)

                # Extract value
                var_val, rest = rest.split(",")
                var_val = int(var_val)
                if var_val not in variables[var_id]:
                    raise ValueError(
                        f"Value {var_val} not declared for variable {var}"
                    )

                # Extract delay
                delay, rest = rest.split(")")
                _, delay = delay.split("-")
                delay = int(delay)
                var_id = var_id + len(variable_names) * (delay - 1)

                body.append(BodyAtom(var_id=var_id, val=var_val, delay=delay))

            r = Rule(head_var_id, head_val, body)
            rules.append(r)

        return LogicProgram(variables, variable_names, rules)

    def logic_form(self) -> str:
        """
        Convert back to its original logic program
        """
        output = ""

        # Variables declaration
        for var, vals in self.variables.items():
            output += "VAR " + self.variable_names[var]
            for val in vals:
                output += " " + str(val)
            output += "\n"

        output += "\n"

        # Rules
        for r in self.rules:
            output += r.to_string(self.variable_names) + "\n"

        return output

    def next(self, state: State) -> State:
        """
        Compute the next state according to the rules of the program,
        assuming a synchronous deterministic semantic.
        """

        # default next value of each variable is 0
        output = [0 for _ in self.variable_names]

        for r in self.rules:
            if output[r.head_var_id] == 1:
                # Add by Kun, cause program is in DNF format (not sure what's
                # going on here?)
                continue
            if r.matches(state):
                output[r.head_var_id] = r.head_val
            else:
                output[r.head_var_id] = 0

        return output

    def generate_transitions(
        self, num_transition_pairs: int
    ) -> list[Transition]:
        """
        Generate randomly a given number of deterministic synchronous
        transitions from the given system.
        """

        def _gen_transition() -> Transition:
            # Random state
            s1 = [random.randint(0, 1) for _ in self.variable_names]
            # Next state according to rules
            s2 = self.next(s1)
            return (s1, s2)

        return [_gen_transition() for _ in range(num_transition_pairs)]

    def generate_all_transitions(self) -> list[Transition]:
        """
        Generate all possible state of the program and their corresponding
        transition
        """
        return [(s1, self.next(s1)) for s1 in self.all_states()]

    def transitions_to_csv(
        self,
        filepath: str | Path,
        transitions: list[tuple[list[int], list[int]]],
    ) -> None:
        """
        Convert a set of transitions to a csv file
        """
        output = ""
        output += ",".join([f"x{i}" for i in range(len(self.variable_names))])
        output += ","
        output += ",".join([f"y{i}" for i in range(len(self.variable_names))])
        output += "\n"

        def _transition_to_csv_str(
            transition: tuple[list[int], list[int]]
        ) -> str:
            return (
                ",".join(map(str, transition[0]))
                + ","
                + ",".join(map(str, transition[1]))
                + "\n"
            )

        output += "".join(map(_transition_to_csv_str, transitions))

        with open(filepath, "w") as f:
            f.write(output)

    def generate_all_time_series(self, length: int) -> list[State]:
        """
        Generate all possible time series of given length produced by the
        logic program:
        all sequence of transitions starting from a state of the program.
        Since all BN programs' delays are 1 (i.e. T-1 in the rule body), we
        can repeatedly apply `next` on the last state to generate the time
        series.
        """
        all_states = self.all_states()
        time_series = []
        for s in all_states:
            series = [s]
            for _ in range(length):
                series.append(self.next(series[-1]))
            time_series.append(series)
        return time_series

    def compare(
        self, other: "LogicProgram"
    ) -> tuple[list[Rule], list[Rule], list[Rule]]:
        """
        Compare the rules of a logic program with another.
        Returns:
            common: rules in common (p1 intersect p2)
            missing: rules missing in the other (p1 - p2)
            over: rules only present in the other (p2 - p1)
        """

        self_rule_set = set(self.rules)
        other_rule_set = set(other.rules)

        common_rules = self_rule_set.intersection(other_rule_set)
        missing_rules = self_rule_set - other_rule_set
        over_rules = other_rule_set - self_rule_set

        return list(common_rules), list(missing_rules), list(over_rules)

    def all_states(self) -> list[State]:
        """
        Compute all possible state of the logic program: all combination of
        variable values
        """
        # Use itertools.product to compute the cartesian product of all
        # variables values
        return list(map(list, product(*self.get_values())))

    @staticmethod
    def precision(
        expected: list[Transition], predicted: list[Transition]
    ) -> float:
        """
        Evaluate prediction precision on deterministic sets of transitions
        Args:
            expected: list of tuple (list of int, list of int)
                originals transitions of a system
            predicted: list of tuple (list of int, list of int)
                predicted transitions of a system

        Returns:
            float in [0,1]
                the error ratio between expected and predicted
        """
        if len(expected) == 0:
            return 1.0

        # Predict each variable for each state
        total = len(expected) * len(expected[0][0])
        error = 0

        for i in range(len(expected)):
            s1, s2 = expected[i]

            for j in range(len(predicted)):
                s1_, s2_ = predicted[j]

                if len(s1) != len(s1_) or len(s2) != len(s2_):
                    raise ValueError("Invalid prediction set")

                if s1 == s1_:
                    for var in range(len(s2)):
                        if s2_[var] != s2[var]:
                            error += 1
                    break

        precision = 1.0 - (error / total)

        return precision

    # def get_variables(self):
    #     return self.variables

    def get_values(self) -> list[list[int]]:
        """
        Get all the possible values of the variables, in the same order as the
        variable IDs
        """
        return [self.variables[i] for i in range(len(self.variables))]

    def get_rules(self) -> list[Rule]:
        return self.rules

    def get_rules_of(self, var: int, val: int) -> list[Rule]:
        """
        Return all the rules that have the same `head_var_id` and `head_val` as
        the args `var` and `val`
        """
        return [
            r for r in self.rules if r.head_var_id == var and r.head_val == val
        ]

    def to_string(self):
        output = "{"
        output += "\nVariables: " + str(sorted(self.variables.keys()))
        output += "\nRules:\n"
        for r in self.rules:
            output += r.to_string() + "\n"
        output += "}"

        return output

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()
