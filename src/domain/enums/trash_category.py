from enum import Enum


class TrashCategory(Enum):
    """L1 Categories - from Trash-Net (6 classes)"""
    CARDBOARD = "cardboard"
    GLASS = "glass"
    METAL = "metal"
    PAPER = "paper"
    PLASTIC = "plastic"
    TRASH = "trash"  # Reject / Non-recyclable


class PlasticType(Enum):
    """L2 Plastic Types - from PlasticNet (Future)"""
    PET = "PET"      # Water bottles
    HDPE = "HDPE"    # Milk bottles, gallons
    PP = "PP"        # Food containers
    PE = "PE"        # Plastic bags


class MetalType(Enum):
    """L2 Metal Types - from MetalNet (Future)"""
    ALUMINUM = "aluminum"  # Soda cans
    STEEL = "steel"        # Food cans


class GlassType(Enum):
    """L2 Glass Types - from GlassNet (Future)"""
    CLEAR = "clear"   # Clear bottles
    BROWN = "brown"   # Brown bottles
    GREEN = "green"   # Green bottles


# Bin mapping for response
BIN_MAPPING = {
    TrashCategory.CARDBOARD: {"number": 1, "label_th": "กระดาษแข็ง", "label_en": "Cardboard"},
    TrashCategory.GLASS: {"number": 2, "label_th": "แก้ว", "label_en": "Glass"},
    TrashCategory.METAL: {"number": 3, "label_th": "โลหะ", "label_en": "Metal"},
    TrashCategory.PAPER: {"number": 4, "label_th": "กระดาษ", "label_en": "Paper"},
    TrashCategory.PLASTIC: {"number": 5, "label_th": "พลาสติก", "label_en": "Plastic"},
    TrashCategory.TRASH: {"number": 6, "label_th": "ขยะทั่วไป", "label_en": "General Waste"},
}


def get_bin_info(category: TrashCategory) -> dict:
    """Get bin information for a category"""
    return BIN_MAPPING.get(category, BIN_MAPPING[TrashCategory.TRASH])
