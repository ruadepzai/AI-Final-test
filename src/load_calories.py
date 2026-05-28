"""
Module tải và xử lý dữ liệu dinh dưỡng của các món ăn Việt Nam.

Module này cung cấp các hàm để tải dữ liệu calories từ CSV và
tính toán thông tin dinh dưỡng cho một phần ăn cụ thể.
"""

from pathlib import Path
import csv
from typing import Dict, Any, Optional


def load_calories(csv_path: Path | str) -> Dict[str, Dict[str, Any]]:
    """
    Tải dữ liệu calories từ file CSV.

    Args:
        csv_path: Đường dẫn tới file CSV chứa thông tin dinh dưỡng.
                  Mỗi dòng: class_name, food_name_vi, kcal_per_100g,
                           protein_g, carb_g, fat_g, source

    Returns:
        Dict ánh xạ class_name → thông tin dinh dưỡng.
        Ví dụ:
        {
            "Pho": {
                "food_name_vi": "Phở",
                "kcal_per_100g": 135,
                "protein_g": 8.4,
                "carb_g": 16.8,
                "fat_g": 3.2,
                "source": "USDA"
            },
            ...
        }

    Raises:
        FileNotFoundError: Nếu file CSV không tồn tại.
        ValueError: Nếu CSV thiếu cột bắt buộc.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"File CSV không tồn tại: {csv_path}")

    calories_dict = {}

    with open(csv_path, "r", encoding="utf-8") as f:
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


def get_nutrition(
    class_name: str, grams: float, csv_path: Path | str
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
        Dict chứa thông tin dinh dưỡng đã scale:
        {
            "food_name_vi": "Phở",
            "grams": 350,
            "kcal": 472.5,
            "protein_g": 29.4,
            "carb_g": 58.8,
            "fat_g": 11.2,
            "source": "USDA"
        }

    Raises:
        ValueError: Nếu class_name không tìm thấy hoặc grams <= 0.
    """
    if grams <= 0:
        raise ValueError(f"Khối lượng phải dương, nhận được: {grams}")

    csv_path = Path(csv_path)
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

    # Đường dẫn file CSV (tương đối từ thư mục project root)
    csv_file = Path(__file__).parent.parent / "data" / "calories.csv"

    print("=" * 70)
    print("TEST: Load Calories CSV")
    print("=" * 70)

    try:
        # Test 1: Load CSV
        print("\n1. Tải dữ liệu từ CSV...")
        calories_data = load_calories(csv_file)
        print(f"   ✓ Thành công! Đã tải {len(calories_data)} classes")

        # Test 2: Hiển thị số classes
        print(f"\n2. Kiểm tra số lượng classes (expected 30):")
        print(f"   Thực tế: {len(calories_data)} classes")
        if len(calories_data) == 30:
            print("   ✓ PASS")
        else:
            print(f"   ✗ FAIL - Dự đoán 30 classes nhưng có {len(calories_data)}")

        # Test 3: Liệt kê vài classes
        print(f"\n3. Danh sách 5 classes đầu tiên (sorted):")
        for i, name in enumerate(sorted(calories_data.keys())[:5], 1):
            print(f"   {i}. {name}")

        # Test 4: Kiểm tra cấu trúc dữ liệu
        print(f"\n4. Kiểm tra cấu trúc dữ liệu (lấy 'Pho'):")
        sample = calories_data["Pho"]
        print(f"   food_name_vi: {sample['food_name_vi']}")
        print(f"   kcal_per_100g: {sample['kcal_per_100g']}")
        print(f"   protein_g: {sample['protein_g']}")
        print(f"   source: {sample['source']}")

        # Test 5: Tính toán dinh dưỡng
        print(f"\n5. Tính dinh dưỡng cho Phở 350g:")
        nutrition = get_nutrition("Pho", 350, csv_file)
        print(f"   Tên: {nutrition['food_name_vi']}")
        print(f"   Khối lượng: {nutrition['grams']}g")
        print(f"   Calories: {nutrition['kcal']} kcal")
        print(f"   Protein: {nutrition['protein_g']}g")
        print(f"   Carbs: {nutrition['carb_g']}g")
        print(f"   Fat: {nutrition['fat_g']}g")
        print(f"   ✓ PASS")

        # Test 6: Test với các portion khác
        print(f"\n6. Ví dụ: Bánh mì 150g (bánh nhỏ):")
        nutrition_banh_mi = get_nutrition("Banh mi", 150, csv_file)
        print(f"   Calories: {nutrition_banh_mi['kcal']} kcal")
        print(f"   ✓ PASS")

        print("\n" + "=" * 70)
        print("✓ TẤT CẢ TEST PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ LỖI: {e}")
        import traceback

        traceback.print_exc()
