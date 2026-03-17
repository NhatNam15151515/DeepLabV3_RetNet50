"""Tầng 3: Ước lượng trọng lượng từ thể tích + mật độ.

Code được trích từ NutritionVerse, dùng bảng mật độ + volume đầu vào
để tính weight theo công thức vật lý m = ρ × V.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, TIER3_CONFIG, NV_REAL_DENSITIES, ensure_directories  # type: ignore


class WeightEstimationResult:
    """Kết quả ước lượng trọng lượng cho một món ăn."""

    def __init__(
        self,
        class_name: str,
        volume_cm3: float,
        density_g_per_cm3: float,
        density_source: str,
    ):
        self.class_name = class_name
        self.volume_cm3 = volume_cm3
        self.density_g_per_cm3 = density_g_per_cm3
        self.density_source = density_source

        # m = ρ × V
        self.weight_grams = volume_cm3 * density_g_per_cm3

        if density_source == "database":
            self.confidence = 0.85
        elif density_source == "manual":
            self.confidence = 0.90
        else:
            self.confidence = 0.60

    def __repr__(self):
        return (
            f"WeightResult("
            f"'{self.class_name}': "
            f"V={self.volume_cm3:.1f}cm³ × "
            f"ρ={self.density_g_per_cm3:.2f}g/cm³ = "
            f"{self.weight_grams:.1f}g "
            f"[{self.density_source}, conf={self.confidence:.0%}])"
        )


class FoodDensityDatabase:
    """Database quản lý mật độ các loại thực phẩm (g/cm³)."""

    def __init__(
        self,
        density_file: Optional[Union[str, Path]] = None,
        default_density: float = 0.9,
    ):
        self.default_density = default_density
        self.density_map: Dict[str, float] = {}
        self.notes_map: Dict[str, str] = {}

        if density_file and Path(density_file).exists():
            self._load_from_csv(density_file)
        else:
            self._init_default_densities()

        # Merge NV-real calibrated densities (ưu tiên cao hơn default)
        if NV_REAL_DENSITIES:
            for k, v in NV_REAL_DENSITIES.items():
                self.density_map[k.strip().lower()] = v
            print(f"[DensityDB] Merged {len(NV_REAL_DENSITIES)} NV-real calibrated densities")

    def _load_from_csv(self, filepath: Union[str, Path]):
        df = pd.read_csv(filepath)

        for _, row in df.iterrows():
            food_type = str(row["food_type"]).strip().lower()
            density = float(row["density_g_per_cm3"])
            notes = str(row.get("notes", ""))

            self.density_map[food_type] = density
            self.notes_map[food_type] = notes

        print(f"[DensityDB] Loaded {len(self.density_map)} food types from {filepath}")

    def _init_default_densities(self):
        """Khởi tạo bảng mật độ mặc định (USDA + tài liệu)."""
        defaults = {
            # Rice & Grains
            "rice": 1.15,
            "fried-rice": 0.85,
            "beef-bowl": 0.95,
            "mixed-rice": 0.95,
            "eels-on-rice": 0.95,
            "pilaf": 0.75,
            "tempura-bowl": 0.85,
            "rice-ball": 1.10,
            "sashimi-bowl": 0.95,
            "sushi-bowl": 0.95,
            "curry-rice": 1.05,

            # Noodles
            "udon-noodle": 0.85,
            "soba-noodle": 0.85,
            "ramen-noodle": 0.90,
            "beef-noodle": 0.95,
            "fried-noodle": 0.85,
            "spaghetti": 0.90,
            "spaghetti-meat-sauce": 0.95,
            "dipping-noodles": 0.85,
            "tempura-udon": 0.88,

            # Breads & Fast Food
            "toast": 0.35,
            "french-fries": 0.45,
            "croissant": 0.25,
            "roll-bread": 0.30,
            "sandwich": 0.55,
            "pizza-toast": 0.50,
            "hot-dog": 0.70,
            "pizza": 0.65,
            "hamburger": 0.65,

            # Soups
            "miso-soup": 1.01,
            "pork-miso-soup": 1.05,
            "chinese-soup": 1.01,
            "stew": 1.05,
            "potage": 1.02,
            "chowder": 1.05,

            # Mains (Meat/Fish)
            "fried-chicken": 0.85,
            "yakitori": 0.95,
            "roast-chicken": 1.00,
            "hambarg-steak": 1.05,
            "beef-steak": 1.08,
            "sweet-and-sour-pork": 1.05,
            "stir-fried-beef-and-peppers": 1.00,
            "fried-fish": 0.85,
            "grilled-salmon": 1.02,
            "sashimi": 1.05,
            "fried-shrimp": 0.80,
            "omelet": 0.90,
            "cold-tofu": 0.95,

            # Sides / Veggies
            "sauteed-vegetables": 0.85,
            "green-salad": 0.35,
            "potato-salad": 1.10,
            "sauteed-spinach": 0.80,
            "macaroni-salad": 1.05,
            "goya-chanpuru": 0.90,
            "vegetable-tempura": 0.60,
        }

        self.density_map = defaults
        print(f"[DensityDB] Initialized with {len(defaults)} default densities")

    def get_density(
        self,
        food_type: str,
        manual_density: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Lấy mật độ cho một loại thực phẩm."""
        if manual_density is not None:
            return (manual_density, "manual")

        food_type_lower = food_type.strip().lower()

        if food_type_lower in self.density_map:
            return (self.density_map[food_type_lower], "database")

        for key, density in self.density_map.items():
            if key in food_type_lower or food_type_lower in key:
                return (density, "database")

        return (self.default_density, "default")

    def add_density(
        self,
        food_type: str,
        density: float,
        notes: str = "",
    ):
        food_type_lower = food_type.strip().lower()
        self.density_map[food_type_lower] = density
        self.notes_map[food_type_lower] = notes

    def save_to_csv(self, filepath: Union[str, Path]):
        data = []
        for food_type, density in self.density_map.items():
            data.append({
                "food_type": food_type,
                "density_g_per_cm3": density,
                "notes": self.notes_map.get(food_type, ""),
            })

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"[DensityDB] Saved {len(data)} entries to {filepath}")


class Tier3WeightEstimation:
    """Module Tier 3: từ volume → weight."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or TIER3_CONFIG

        density_file = self.config.get("density_file")
        default_density = self.config.get("default_density", 0.9)

        self.density_db = FoodDensityDatabase(
            density_file=density_file,
            default_density=default_density,
        )

        self.scale_factor = self.config.get("global_scale_factor", 1.0)

    def estimate_weight(
        self,
        class_name: str,
        volume_cm3: float,
        manual_density: Optional[float] = None,
    ) -> WeightEstimationResult:
        """Ước lượng trọng lượng cho một món."""
        density, source = self.density_db.get_density(class_name, manual_density)

        result = WeightEstimationResult(
            class_name=class_name,
            volume_cm3=volume_cm3 * self.scale_factor,
            density_g_per_cm3=density,
            density_source=source,
        )

        return result

    def estimate_weights_batch(
        self,
        items: List[Tuple[str, float]],
        manual_densities: Optional[Dict[str, float]] = None,
    ) -> List[WeightEstimationResult]:
        """Batch inference nhiều món."""
        manual_densities = manual_densities or {}
        results = []

        for class_name, volume in items:
            manual_d = manual_densities.get(class_name)
            result = self.estimate_weight(class_name, volume, manual_d)
            results.append(result)

        return results

    def calibrate_with_ground_truth(
        self,
        predictions: List[WeightEstimationResult],
        ground_truth_weights: List[float],
    ) -> float:
        """Calibrate scale factor từ ground truth."""
        if len(predictions) != len(ground_truth_weights):
            raise ValueError("Số lượng predictions và ground truth không khớp")

        pred_weights = np.array([p.weight_grams for p in predictions])
        gt_weights = np.array(ground_truth_weights)

        new_scale = np.sum(gt_weights * pred_weights) / (np.sum(pred_weights ** 2) + 1e-8)

        print(f"[Tier3] Calibration: old_scale={self.scale_factor:.4f} -> new_scale={new_scale:.4f}")
        self.scale_factor = new_scale

        return new_scale

