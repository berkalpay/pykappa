import random
from typing import Any, Optional, Iterable, Generic, TypeVar, Self
from collections import defaultdict
from collections.abc import Callable, Hashable


def str_table(rows: list[list], header: Optional[list] = None) -> str:
    """Format rows into a table with aligned columns.

    Args:
        rows: List of data rows
        header: Optional header row

    Returns:
        Formatted table string
    """
    all_rows = [header] + rows if header else rows

    num_cols = len(all_rows[0])
    col_widths = [max(len(str(row[i])) for row in all_rows) for i in range(num_cols)]

    formatted_rows = []
    for i, row in enumerate(all_rows):
        formatted_rows.append(
            " | ".join(f"{str(item):<{col_widths[j]}}" for j, item in enumerate(row))
        )
        if i == 0 and header:
            formatted_rows.append("-" * len(formatted_rows[-1]))

    return "\n".join(formatted_rows)


def rejection_sample(population: Iterable, excluded: Iterable, max_attempts: int = 100):
    population = list(population)
    if not population:
        raise ValueError("Sequence is empty")
    excluded_ids = set(id(x) for x in excluded)

    # Fast rejection sampling (O(1) average case for small exclusion sets)
    for _ in range(max_attempts):
        choice = random.choice(population)
        if id(choice) not in excluded_ids:
            return choice

    # Fallback to O(n) scan only if necessary (rare for small exclusion sets)
    valid_choices = [item for item in population if id(item) not in excluded_ids]
    if not valid_choices:
        raise ValueError("No valid elements to choose from")
    return random.choice(valid_choices)


class OrderedSet[T]:
    def __init__(self, items: Optional[Iterable[T]] = None):
        self.dict = dict() if items is None else dict.fromkeys(items)

    def __iter__(self):
        yield from self.dict

    def __len__(self):
        return len(self.dict)

    def add(self, item: Any) -> None:
        self.dict[item] = None

    def remove(self, item: Any) -> None:
        del self.dict[item]


class Counted:
    counter = 0

    def __init__(self):
        self.id = Counted.counter
        Counted.counter += 1

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return hash(self) == hash(other)


T = TypeVar("T")  # Member type of `IndexedSet`
Property = Callable[
    [T], Iterable[Hashable]
]  # Callable that returns the property values of an item


class IndexedSet(set[T], Generic[T]):
    """
    A subclass of the built-in `set`, with support for indexing
    by arbitrary properties of set members, as well as integer
    indexing to allow for random sampling.

    Credit https://stackoverflow.com/a/15993515 for the integer indexing logic.

    NOTE: Although this class is indexable, member ordering is not stable
    across insertions and deletions.

    Example usage:
    ```
    @dataclass
    class SportsTeam:
        name: str
        jersey_color: str
        members: list[str]

    teams: IndexedSet[SportsTeam] = IndexedSet()
    teams.create_index("name", lambda team: [team.name])
    teams.create_index("color", lambda team: [team.jersey_color])

    [...] # populate the set with teams

    teams.lookup_one("name", "Manchester") # Returns the team whose name is "Manchester"
    teams.lookup("color", "blue") # Returns all teams with blue jerseys
    ```
    """

    _item_to_pos: dict[T, int]
    _item_list: list[T]
    properties: dict[str, Property]
    indices: dict[str, defaultdict[Hashable, Self]]

    def __init__(self, iterable: Iterable[T] = []):
        iterable = list(iterable)
        super().__init__(iterable)

        self._item_list = iterable
        self._item_to_pos = {item: i for (i, item) in enumerate(iterable)}
        self.properties = {}
        self.indices = {}

    def __getitem__(self, i):
        assert 0 <= i < len(self)
        return self._item_list[i]

    def add(self, item: T):
        if item in self:
            return
        super().add(item)

        self._item_list.append(item)
        self._item_to_pos[item] = len(self._item_list) - 1
        self._update_indices_for_item(item, adding=True)

    def remove(self, item: T):
        assert item in self
        super().remove(item)

        pos = self._item_to_pos.pop(item)
        last_item = self._item_list.pop()
        if pos < len(self._item_list):
            self._item_list[pos] = last_item
            self._item_to_pos[last_item] = pos

        self._update_indices_for_item(item, adding=False)

    def _update_indices_for_item(self, item: T, adding: bool):
        """Update property indices when adding or removing an item."""
        for prop_name, prop in self.properties.items():
            for val in prop(item):
                index = self.indices[prop_name][val]

                if adding:
                    index.add(item)
                else:
                    index.remove(item)
                    if not index:
                        del self.indices[prop_name][val]

    def lookup(self, name: str, value: Any) -> Self:
        """Returns an IndexedSet of all matching items."""
        return self.indices[name][value]

    def lookup_one(self, name: str, value: Any) -> T:
        """Returns a single matching item. Raises if not exactly one match."""
        matches = self.indices[name][value]
        assert len(matches) == 1
        return next(iter(matches))

    def remove_by(self, prop_name: str, value: Any):
        """Remove all set members whose given property matches `value`."""
        if value in self.indices[prop_name]:
            for match in list(self.indices[prop_name][value]):
                assert match in self
                self.remove(match)

    def create_index(self, name: str, prop: Property):
        """
        By the given property, create an index that's updated when adding and removing members.

        Args:
            name: Name of the index
            prop: A callable that returns an iterable of hashable values of the item

        NOTE: Mutating set members outside of interface calls can invalidate indices.
        """
        assert name not in self.properties
        self.properties[name] = prop
        self.indices[name] = defaultdict(IndexedSet)

        for el in self:
            for val in prop(el):
                self.indices[name][val].add(el)
