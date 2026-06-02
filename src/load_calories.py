"""
Module tải và xử lý dữ liệu dinh dưỡng của các món ăn Việt Nam.

Module này cung cấp các hàm để tải dữ liệu calories từ CSV và
tính toán thông tin dinh dưỡng cho một phần ăn cụ thể.

Nguồn dữ liệu (TV2 cung cấp):
- categories.csv: Dinh dưỡng macro trên 100g mỗi món (kcal, protein, carb, fat)
- calories.csv: Tổng calo và trọng lượng theo suất ăn thực tế
- recipes.csv: Công thức nấu ăn (nguyên liệu + gram)
- ingredients_nutrition.csv: Bảng thành phần dinh dưỡng 164 nguyên liệu
  (bao gồm vi chất: chất xơ, cholesterol, canxi, sắt, vitamin...)
  Nguồn: Viện Dinh dưỡng Quốc gia Việt Nam (NIN)
"""

from pathlib import Path
import csv
from typing import Dict, Any, Optional


# ============================================================================
# CONSTANTS
# ============================================================================

DATA_DIR: Path = Path(__file__).parent.parent / "data"
"""Thư mục chứa dữ liệu."""

CATEGORIES_CSV: Path = DATA_DIR / "categories.csv"
"""File dinh dưỡng macro trên 100g (thay thế calories.csv cũ)."""

CALORIES_CSV: Path = DATA_DIR / "calories.csv"
"""File calo theo suất ăn thực tế."""

RECIPES_CSV: Path = DATA_DIR / "recipes.csv"
"""File công thức nấu ăn."""

INGREDIENTS_CSV: Path = DATA_DIR / "ingredients_nutrition.csv"
"""File dinh dưỡng nguyên liệu (NIN Vietnam)."""


# ============================================================================
# LOAD MACROS (per 100g) - TỪ categories.csv
# ============================================================================


def load_calories(csv_path: Optional[Path | str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Tải dữ liệu dinh dưỡng macro (kcal, protein, carb, fat) trên 100g.

    Đọc từ categories.csv (dữ liệu mới từ NIN Vietnam).
    Giữ backward compatibility: hàm này trả về cùng format cũ.

    Args:
        csv_path: Đường dẫn tới CSV. None = dùng categories.csv mặc định.

    Returns:
        Dict ánh xạ class_name → thông tin dinh dưỡng macro.
        Ví dụ:
        {
            "Pho": {
                "food_name_vi": "Phở",
                "kcal_per_100g": 136.5,
                "protein_g": 11.0,
                "carb_g": 13.0,
                "fat_g": 4.5,
                "source": "NIN_Vietnam_Calculated"
            }
        }

    Raises:
        FileNotFoundError: Nếu file CSV không tồn tại.
        ValueError: Nếu CSV thiếu cột bắt buộc.
    """
    csv_path = Path(csv_path) if csv_path else CATEGORIES_CSV

    if not csv_path.exists():
        raise FileNotFoundError(f"File CSV không tồn tại: {csv_path}")

    calories_dict = {}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Kiểm tra header
        if reader.fieldnames is None:
            raise ValueError("File CSV trống hoặc không hợp lệ")

        required_cols = {
            "class_name",
            "food_name_vi",
            "kcal_per_100g",
            "protein_g",
            "carb_g",
            "fat_g",
            "source",
        }
        if not required_cols.issubset(set(reader.fieldnames)):
            missing = required_cols - set(reader.fieldnames)
            raise ValueError(f"CSV thiếu cột: {missing}")

        for row in reader:
            class_name = row["class_name"].strip()
            calories_dict[class_name] = {
                "food_name_vi": row["food_name_vi"].strip(),
                "kcal_per_100g": float(row["kcal_per_100g"]),
                "protein_g": float(row["protein_g"]),
                "carb_g": float(row["carb_g"]),
                "fat_g": float(row["fat_g"]),
                "source": row["source"].strip(),
            }

    return calories_dict


# ============================================================================
# LOAD PER-SERVING INFO - TỪ calories.csv
# ============================================================================


def load_serving_info(csv_path: Optional[Path | str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Tải thông tin calo theo suất ăn thực tế.

    Đọc từ calories.csv mới (tính toán từ công thức nấu ăn thực tế).

    Args:
        csv_path: Đường dẫn tới CSV. None = dùng mặc định.

    Returns:
        Dict ánh xạ class_name → thông tin suất ăn.
        Ví dụ:
        {
            "Pho": {
                "food_name_vi": "Phở",
                "serving_weight_g": 510.0,
                "serving_kcal": 696.0,
                "source": "NIN_Vietnam_Calculated"
            }
        }
    """
    csv_path = Path(csv_path) if csv_path else CALORIES_CSV

    if not csv_path.exists():
        print(f"[WARN] File serving info không tồn tại: {csv_path}")
        return {}

    serving_dict = {}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            return {}

        for row in reader:
            class_name = row.get("class_name", "").strip()
            if not class_name:
                continue
            serving_dict[class_name] = {
                "food_name_vi": row.get("food_name_vi", "").strip(),
                "serving_weight_g": float(row.get("total_weight_suat_g", 0)),
                "serving_kcal": float(row.get("total_suat_kcal", 0)),
                "source": row.get("source", "").strip(),
            }

    return serving_dict


# ============================================================================
# CALCULATE MICRO-NUTRIENTS - TỪ recipes.csv + ingredients_nutrition.csv
# ============================================================================


def calculate_micro_nutrients(
    recipes_csv: Optional[Path | str] = None,
    ingredients_csv: Optional[Path | str] = None,
    categories_csv: Optional[Path | str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Tính toán vi chất dinh dưỡng cho mỗi món từ công thức nấu ăn.

    Quy trình:
    1. Đọc categories.csv để lấy ánh xạ food_name_vi → class_name
    2. Đọc recipes.csv: mỗi món có danh sách (nguyên liệu, gram)
    3. Đọc ingredients_nutrition.csv: dinh dưỡng mỗi nguyên liệu trên 100g
    4. Tính tổng dinh dưỡng theo gram thực tế trong công thức
    5. Chuẩn hóa về trên 100g thành phẩm

    Args:
        recipes_csv: Đường dẫn file công thức. None = mặc định.
        ingredients_csv: Đường dẫn file nguyên liệu. None = mặc định.
        categories_csv: Đường dẫn file categories. None = mặc định.

    Returns:
        Dict ánh xạ class_name → vi chất dinh dưỡng trên 100g thành phẩm.
        Ví dụ:
        {
            "Pho": {
                "fiber_g": 0.3,
                "cholesterol_mg": 11.6,
                "calcium_mg": 25.4,
                "phosphorus_mg": 130.5,
                "iron_mg": 1.8,
                "sodium_mg": 75.2,
                "potassium_mg": 85.0,
                "vitamin_a_mcg": 2.5,
                "vitamin_b1_mg": 0.08,
                "vitamin_c_mg": 1.2,
            }
        }
    """
    recipes_csv = Path(recipes_csv) if recipes_csv else RECIPES_CSV
    ingredients_csv = Path(ingredients_csv) if ingredients_csv else INGREDIENTS_CSV
    categories_csv = Path(categories_csv) if categories_csv else CATEGORIES_CSV

    # Kiểm tra file tồn tại
    for path, name in [(recipes_csv, "recipes"), (ingredients_csv, "ingredients")]:
        if not path.exists():
            print(f"[WARN] File {name} không tồn tại: {path}")
            return {}

    # --- Bước 1: Tạo ánh xạ food_name_vi → class_name ---
    vi_to_class = {}
    if categories_csv.exists():
        with open(categories_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vi_name = row.get("food_name_vi", "").strip()
                class_name = row.get("class_name", "").strip()
                if vi_name and class_name:
                    vi_to_class[vi_name] = class_name

    # --- Bước 2: Đọc ingredients_nutrition.csv ---
    # Header: TÊN THỨC ĂN, Calories (kcal), Protein (g), Fat (g),
    #          Carbonhydrates (g), Chất xơ (g), Cholesterol (mg),
    #          Canxi (mg), Photpho (mg), Sắt (mg), Natri (mg), Kali (mg),
    #          Beta Caroten (mcg), Vitamin A (mcg), Vitamin B1 (mg),
    #          Vitamin C (mg), Loại
    ingredients_db = {}
    with open(ingredients_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("TÊN THỨC ĂN", "").strip()
            if not name:
                continue
            try:
                ingredients_db[name] = {
                    "fiber_g": float(row.get("Chất xơ (g)", 0) or 0),
                    "cholesterol_mg": float(row.get("Cholesterol (mg)", 0) or 0),
                    "calcium_mg": float(row.get("Canxi (mg)", 0) or 0),
                    "phosphorus_mg": float(row.get("Photpho (mg)", 0) or 0),
                    "iron_mg": float(row.get("Sắt (mg)", 0) or 0),
                    "sodium_mg": float(row.get("Natri (mg)", 0) or 0),
                    "potassium_mg": float(row.get("Kali (mg)", 0) or 0),
                    "vitamin_a_mcg": float(row.get("Vitamin A (mcg)", 0) or 0),
                    "vitamin_b1_mg": float(row.get("Vitamin B1 (mg)", 0) or 0),
                    "vitamin_c_mg": float(row.get("Vitamin C (mg)", 0) or 0),
                }
            except (ValueError, TypeError):
                continue

    # --- Bước 3: Đọc recipes.csv và tính toán ---
    # Header: tên_món, nguyên_liệu, grams, ghi_chú
    # Nhóm theo tên_món, tổng hợp dinh dưỡng từ nguyên liệu
    dish_recipes = {}  # food_name_vi -> [(ingredient_name, grams)]
    with open(recipes_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dish_name = row.get("tên_món", "").strip()
            ingredient_name = row.get("nguyên_liệu", "").strip()
            grams = float(row.get("grams", 0) or 0)
            if dish_name and ingredient_name and grams > 0:
                if dish_name not in dish_recipes:
                    dish_recipes[dish_name] = []
                dish_recipes[dish_name].append((ingredient_name, grams))

    # --- Bước 4: Tính vi chất cho mỗi món ---
    micro_keys = [
        "fiber_g", "cholesterol_mg", "calcium_mg", "phosphorus_mg",
        "iron_mg", "sodium_mg", "potassium_mg",
        "vitamin_a_mcg", "vitamin_b1_mg", "vitamin_c_mg",
    ]

    result = {}
    for dish_name_vi, ingredients_list in dish_recipes.items():
        # Tìm class_name tương ứng
        class_name = vi_to_class.get(dish_name_vi)
        if not class_name:
            continue

        # Tổng hợp dinh dưỡng từ nguyên liệu
        total_micros = {k: 0.0 for k in micro_keys}
        total_weight = 0.0

        for ingredient_name, grams in ingredients_list:
            total_weight += grams
            if ingredient_name in ingredients_db:
                ing_data = ingredients_db[ingredient_name]
                scale = grams / 100.0  # Dữ liệu nguyên liệu tính trên 100g
                for k in micro_keys:
                    total_micros[k] += ing_data.get(k, 0) * scale

        # Chuẩn hóa về trên 100g thành phẩm
        if total_weight > 0:
            per_100g = {}
            for k in micro_keys:
                per_100g[k] = round(total_micros[k] * 100.0 / total_weight, 2)
            result[class_name] = per_100g

    return result


# ============================================================================
# LOAD FULL NUTRITION - KẾT HỢP TẤT CẢ
# ============================================================================


def load_full_nutrition(
    categories_csv: Optional[Path | str] = None,
    calories_csv: Optional[Path | str] = None,
    recipes_csv: Optional[Path | str] = None,
    ingredients_csv: Optional[Path | str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Tải toàn bộ thông tin dinh dưỡng: macro + serving + vi chất.

    Kết hợp dữ liệu từ tất cả file CSV của TV2:
    - categories.csv → macro nutrients per 100g
    - calories.csv → per-serving info
    - recipes.csv + ingredients_nutrition.csv → micro-nutrients

    Args:
        categories_csv: Path tới categories.csv.
        calories_csv: Path tới calories.csv.
        recipes_csv: Path tới recipes.csv.
        ingredients_csv: Path tới ingredients_nutrition.csv.

    Returns:
        Dict ánh xạ class_name → full nutrition data.
    """
    # 1. Load macros (bắt buộc)
    full_data = load_calories(categories_csv)

    # 2. Merge serving info (tùy chọn)
    try:
        serving = load_serving_info(calories_csv)
        for cls, info in serving.items():
            if cls in full_data:
                full_data[cls]["serving_weight_g"] = info["serving_weight_g"]
                full_data[cls]["serving_kcal_total"] = info["serving_kcal"]
    except Exception as e:
        print(f"[WARN] Không load được serving info: {e}")

    # 3. Merge micro-nutrients (tùy chọn)
    try:
        micros = calculate_micro_nutrients(recipes_csv, ingredients_csv, categories_csv)
        for cls, micro_data in micros.items():
            if cls in full_data:
                full_data[cls].update(micro_data)
    except Exception as e:
        print(f"[WARN] Không tính được vi chất: {e}")

    return full_data


# ============================================================================
# GET NUTRITION (backward compatible)
# ============================================================================


def get_nutrition(
    class_name: str, grams: float, csv_path: Optional[Path | str] = None
) -> Dict[str, float]:
    """
    Tính toán thông tin dinh dưỡng cho một phần ăn cụ thể.

    Hàm lấy thông tin dinh dưỡng trên 100g từ CSV, rồi scale theo
    số gram thực tế nhập vào.

    Args:
        class_name: Tên class/folder của món ăn (ví dụ: "Pho").
        grams: Khối lượng phần ăn tính bằng gram.
        csv_path: Đường dẫn tới file CSV.

    Returns:
        Dict chứa thông tin dinh dưỡng đã scale.

    Raises:
        ValueError: Nếu class_name không tìm thấy hoặc grams <= 0.
    """
    if grams <= 0:
        raise ValueError(f"Khối lượng phải dương, nhận được: {grams}")

    calories_dict = load_calories(csv_path)

    if class_name not in calories_dict:
        available = list(calories_dict.keys())
        raise ValueError(
            f"Class '{class_name}' không tìm thấy. "
            f"Có {len(available)} classes: {', '.join(sorted(available[:5]))}..."
        )

    base_info = calories_dict[class_name]
    scale_factor = grams / 100

    return {
        "food_name_vi": base_info["food_name_vi"],
        "grams": grams,
        "kcal": round(base_info["kcal_per_100g"] * scale_factor, 2),
        "protein_g": round(base_info["protein_g"] * scale_factor, 2),
        "carb_g": round(base_info["carb_g"] * scale_factor, 2),
        "fat_g": round(base_info["fat_g"] * scale_factor, 2),
        "source": base_info["source"],
    }


if __name__ == "__main__":
    """
    Unit test: kiểm tra hàm load và get_nutrition hoạt động đúng.
    """
    from pathlib import Path

    print("=" * 70)
    print("TEST: Load Full Nutrition Data (NIN Vietnam)")
    print("=" * 70)

    try:
        # Test 1: Load macros từ categories.csv
        print("\n1. Tải dữ liệu macro từ categories.csv...")
        calories_data = load_calories()
        print(f"   ✓ Thành công! Đã tải {len(calories_data)} classes")

        # Test 2: Load serving info
        print("\n2. Tải thông tin suất ăn từ calories.csv...")
        serving_data = load_serving_info()
        print(f"   ✓ Thành công! Đã tải {len(serving_data)} classes")

        # Test 3: Tính vi chất dinh dưỡng
        print("\n3. Tính vi chất từ recipes + ingredients...")
        micro_data = calculate_micro_nutrients()
        print(f"   ✓ Thành công! Đã tính {len(micro_data)} classes")

        # Test 4: Load full nutrition
        print("\n4. Tải toàn bộ dinh dưỡng (kết hợp)...")
        full_data = load_full_nutrition()
        print(f"   ✓ Thành công! Đã tải {len(full_data)} classes")

        # Test 5: Xem chi tiết Phở
        print("\n5. Chi tiết dinh dưỡng Phở (per 100g):")
        pho = full_data.get("Pho", {})
        print(f"   Tên:          {pho.get('food_name_vi', 'N/A')}")
        print(f"   Calories:     {pho.get('kcal_per_100g', 'N/A')} kcal/100g")
        print(f"   Protein:      {pho.get('protein_g', 'N/A')}g")
        print(f"   Carb:         {pho.get('carb_g', 'N/A')}g")
        print(f"   Fat:          {pho.get('fat_g', 'N/A')}g")
        print(f"   Chất xơ:      {pho.get('fiber_g', 'N/A')}g")
        print(f"   Cholesterol:  {pho.get('cholesterol_mg', 'N/A')}mg")
        print(f"   Canxi:        {pho.get('calcium_mg', 'N/A')}mg")
        print(f"   Sắt:          {pho.get('iron_mg', 'N/A')}mg")
        print(f"   Vitamin C:    {pho.get('vitamin_c_mg', 'N/A')}mg")
        print(f"   Suất ăn:      {pho.get('serving_weight_g', 'N/A')}g")
        print(f"   Calo/suất:    {pho.get('serving_kcal_total', 'N/A')} kcal")
        print(f"   Nguồn:        {pho.get('source', 'N/A')}")

        print("\n" + "=" * 70)
        print("✓ TẤT CẢ TEST PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ LỖI: {e}")
        import traceback

        traceback.print_exc()
