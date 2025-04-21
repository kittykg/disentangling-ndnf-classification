from enum import Enum
from pathlib import Path


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

MUTATION_RATIO_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
INCOMPLETE_RATIO_LIST = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

LOGIC_PROGRAM_DIR = root / "data" / "logic_program"



class BNDatasetType(Enum):
    ARA = "ara"
    BUDDING = "budding"
    FISSION = "fission"
    MAM = "mam"
    SAMPLE = "sample"

    @staticmethod
    def from_str(s: str) -> "BNDatasetType":
        if s not in [e.value for e in BNDatasetType]:
            raise ValueError(f"Unsupported dataset type: {s}")

        return BNDatasetType[s.upper()]


class BNDatasetSubType(Enum):
    NORMAL = "normal"
    FUZZY = "fuzzy"
    INCOMPLETE = "incomplete"

    @staticmethod
    def from_str(s: str) -> "BNDatasetSubType":
        if s not in [e.value for e in BNDatasetSubType]:
            raise ValueError(f"Unsupported dataset subtype: {s}")

        return BNDatasetSubType[s.upper()]

    def subtype_folder_prefix(self) -> str:
        if self == BNDatasetSubType.NORMAL:
            return ""
        if self == BNDatasetSubType.FUZZY:
            return "mr"
        return "ir"
