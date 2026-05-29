"""
Recipe Calculator - Tính toán calo cho các món ăn
Cho phép nhập nguyên liệu tùy chỉnh và tính calo tự động
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import sys
from pathlib import Path

# Tự động tìm đường dẫn thư mục gốc (AI-Final-test) và thêm vào hệ thống
workspace_root = str(Path(__file__).parent.parent)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.ingredients import get_ingredients_db
from src.recipes import get_recipes_db


class RecipeCalculator:
    """Công cụ tính calo cho các công thức"""
    
    def __init__(self):
        """Khởi tạo calculator với databases"""
        self.ingredients_db = get_ingredients_db()
        self.recipes_db = get_recipes_db()
    
    def calculate_ingredient_nutrition(self, ingredient_name: str, grams: float) -> Optional[Dict]:
        """
        Tính dinh dưỡng cho 1 nguyên liệu
        
        Args:
            ingredient_name: Tên nguyên liệu
            grams: Lượng gam
        
        Returns:
            Dict: {kcal, protein_g, carb_g, fat_g}
        """
        return self.ingredients_db.get_nutrition(ingredient_name, grams)
    
    def calculate_dish_calories(self, dish_name: str, num_servings: float = 1.0) -> Optional[Dict]:
        """
        Tính calo cho 1 dish (từ recipe database)
        
        Args:
            dish_name: Tên món ăn
            num_servings: Số phần (1 = 1 bát, 2 = 2 bát, v.v...)
        
        Returns:
            Dict: Dữ liệu tổng hợp cấu trúc không đổi
        """
        recipe = self.recipes_db.get_recipe(dish_name)
        if recipe is None:
            return None
        
        total_kcal = 0
        total_protein = 0
        total_carb = 0
        total_fat = 0
        
        ingredients_breakdown = []
        
        for ingredient in recipe:
            # SỬA ĐỔI TÊN TIÊU ĐỀ: Giữ nguyên cấu trúc key nội bộ tiếng Anh, chỉ thay đổi
            # từ khóa lấy dữ liệu từ Dictionary của RecipesDB cho khớp tiếng Việt
            ing_name = ingredient['ingredient_name'] # key trong list vẫn giữ nguyên như cũ
            grams = ingredient['grams']
            notes = ingredient['notes']
            
            nutrition = self.calculate_ingredient_nutrition(ing_name, grams)
            if nutrition is None:
                continue
            
            kcal = nutrition['kcal']
            protein = nutrition['protein_g']
            carb = nutrition['carb_g']
            fat = nutrition['fat_g']
            
            total_kcal += kcal
            total_protein += protein
            total_carb += carb
            total_fat += fat
            
            ing_db = self.ingredients_db.get_ingredient(ing_name)
            ing_vi = ing_db['name_vi'] if ing_db else ing_name
            
            ingredients_breakdown.append({
                'name': ing_name,
                'name_vi': ing_vi,
                'grams': grams,
                'kcal': kcal,
                'protein_g': protein,
                'carb_g': carb,
                'fat_g': fat,
                'notes': notes
            })
        
        return {
            'dish_name': dish_name,
            'num_servings': num_servings,
            'total_kcal': total_kcal * num_servings,
            'per_serving_kcal': total_kcal,
            'total_protein_g': total_protein * num_servings,
            'total_carb_g': total_carb * num_servings,
            'total_fat_g': total_fat * num_servings,
            'per_serving_protein_g': total_protein,
            'per_serving_carb_g': total_carb,
            'per_serving_fat_g': total_fat,
            'ingredients': ingredients_breakdown
        }
    
    def calculate_custom_recipe(self, ingredients_dict: Dict[str, float], recipe_name: str = "Custom") -> Dict:
        """
        Tính calo cho công thức tùy chỉnh
        
        Args:
            ingredients_dict: {'Beef': 100, 'Rice': 50, ...}
            recipe_name: Tên công thức
        
        Returns:
            Dict với thông tin calo tương tự calculate_dish_calories
        """
        total_kcal = 0
        total_protein = 0
        total_carb = 0
        total_fat = 0
        
        ingredients_breakdown = []
        
        for ing_name, grams in ingredients_dict.items():
            nutrition = self.calculate_ingredient_nutrition(ing_name, grams)
            if nutrition is None:
                print(f"⚠️ Warning: Ingredient '{ing_name}' not found!")
                continue
            
            kcal = nutrition['kcal']
            protein = nutrition['protein_g']
            carb = nutrition['carb_g']
            fat = nutrition['fat_g']
            
            total_kcal += kcal
            total_protein += protein
            total_carb += carb
            total_fat += fat
            
            ing_db = self.ingredients_db.get_ingredient(ing_name)
            ing_vi = ing_db['name_vi'] if ing_db else ing_name
            
            ingredients_breakdown.append({
                'name': ing_name,
                'name_vi': ing_vi,
                'grams': grams,
                'kcal': kcal,
                'protein_g': protein,
                'carb_g': carb,
                'fat_g': fat,
                'notes': 'Custom'
            })
        
        return {
            'dish_name': recipe_name,
            'num_servings': 1,
            'total_kcal': total_kcal,
            'per_serving_kcal': total_kcal,
            'total_protein_g': total_protein,
            'total_carb_g': total_carb,
            'total_fat_g': total_fat,
            'per_serving_protein_g': total_protein,
            'per_serving_carb_g': total_carb,
            'per_serving_fat_g': total_fat,
            'ingredients': ingredients_breakdown
        }
    
    def get_detailed_breakdown(self, dish_name: str) -> Optional[pd.DataFrame]:
        """
        Lấy chi tiết breakdown từng thành phần
        
        Args:
            dish_name: Tên món ăn
        
        Returns:
            DataFrame với các cột: name, grams, kcal, protein_g, carb_g, fat_g, % calo
        """
        result = self.calculate_dish_calories(dish_name)
        if result is None:
            return None
        
        df = pd.DataFrame(result['ingredients'])
        total_kcal = result['per_serving_kcal']
        
        if total_kcal > 0:
            df['kcal_percent'] = (df['kcal'] / total_kcal * 100).round(1)
        else:
            df['kcal_percent'] = 0
        
        return df[['name', 'name_vi', 'grams', 'kcal', 'kcal_percent', 'protein_g', 'carb_g', 'fat_g', 'notes']]
    
    def compare_dishes(self, dish_names: List[str]) -> pd.DataFrame:
        """
        So sánh calo của nhiều dishes
        
        Args:
            dish_names: ['Pho', 'Banh mi', ...]
        
        Returns:
            DataFrame so sánh
        """
        data = []
        for dish in dish_names:
            result = self.calculate_dish_calories(dish)
            if result:
                data.append({
                    'Dish': dish,
                    'Calo': result['per_serving_kcal'],
                    'Protein (g)': result['per_serving_protein_g'],
                    'Carb (g)': result['per_serving_carb_g'],
                    'Fat (g)': result['per_serving_fat_g'],
                })
        
        return pd.DataFrame(data)
    
    def print_dish_nutrition(self, dish_name: str, num_servings: float = 1.0):
        """In thông tin calo dish dễ đọc"""
        result = self.calculate_dish_calories(dish_name, num_servings)
        if result is None:
            print(f"❌ Dish '{dish_name}' not found!")
            return
        
        print(f"\n{'='*70}")
        print(f"🍲 {dish_name.upper()}")
        if num_servings > 1:
            print(f"× {num_servings:.1f} SERVINGS")
        print(f"{'='*70}")
        
        print(f"\n📊 NUTRITION SUMMARY:")
        print(f"  Calories:  {result['total_kcal']:>8.1f} kcal")
        print(f"  Protein:   {result['total_protein_g']:>8.1f} g")
        print(f"  Carbs:     {result['total_carb_g']:>8.1f} g")
        print(f"  Fat:       {result['total_fat_g']:>8.1f} g")
        
        if num_servings == 1:
            print(f"\n📋 INGREDIENTS BREAKDOWN:")
            print(f"{'─'*70}")
            print(f"{'Ingredient':<20} {'Grams':>8} {'Calo':>8} {'%':>6} {'Protein':>8} {'Carb':>8} {'Fat':>8}")
            print(f"{'─'*70}")
            
            for ing in result['ingredients']:
                total_kcal = result['per_serving_kcal']
                pct = (ing['kcal'] / total_kcal * 100) if total_kcal > 0 else 0
                print(f"{ing['name']:<20} {ing['grams']:>8.0f} {ing['kcal']:>8.1f} {pct:>5.1f}% {ing['protein_g']:>8.1f} {ing['carb_g']:>8.1f} {ing['fat_g']:>8.1f}")
            
            print(f"{'─'*70}")
            print(f"{'TOTAL':<20} {sum(i['grams'] for i in result['ingredients']):>8.0f} {result['per_serving_kcal']:>8.1f} {100.0:>5.1f}% {result['per_serving_protein_g']:>8.1f} {result['per_serving_carb_g']:>8.1f} {result['per_serving_fat_g']:>8.1f}")
        
        print(f"{'='*70}\n")
    
    def print_comparison(self, dish_names: List[str]):
        """In bảng so sánh dishes"""
        df = self.compare_dishes(dish_names)
        
        print(f"\n{'='*70}")
        print(f"📊 DISHES COMPARISON")
        print(f"{'='*70}\n")
        print(df.to_string(index=False))
        
        highest_kcal_idx = df['Calo'].idxmax()
        lowest_kcal_idx = df['Calo'].idxmin()
        
        print(f"\n{'─'*70}")
        print(f"Highest calo: {df.iloc[highest_kcal_idx]['Dish']} ({df.iloc[highest_kcal_idx]['Calo']:.1f} kcal)")
        print(f"Lowest calo:  {df.iloc[lowest_kcal_idx]['Dish']} ({df.iloc[lowest_kcal_idx]['Calo']:.1f} kcal)")
        print(f"Difference:   {df.iloc[highest_kcal_idx]['Calo'] - df.iloc[lowest_kcal_idx]['Calo']:.1f} kcal")
        print(f"{'='*70}\n")

    def export_all_to_csv(self):
        """Tự động quét toàn bộ món ăn trong recipes và xuất ra 2 file CSV kết quả"""
        all_dishes = self.recipes_db.get_all_dishes()
        categories_records = []
        
        mapping_class = {
            'Bánh bèo': 'Banh beo', 'Bánh bột lọc': 'Banh bot loc', 'Bánh căn': 'Banh can',
            'Bánh canh': 'Banh canh', 'Bánh chưng': 'Banh chung', 'Bánh cuốn': 'Banh cuon',
            'Bánh đúc': 'Banh duc', 'Bánh giò': 'Banh gio', 'Bánh khọt': 'Banh khot',
            'Bánh mì': 'Banh mi', 'Bánh pía': 'Banh pia', 'Bánh tét': 'Banh tet',
            'Bánh tráng nướng': 'Banh trang nuong', 'Bánh xèo': 'Banh xeo', 'Bún bò Huế': 'Bun bo Hue',
            'Bún đậu mắm tôm': 'Bun dau mam tom', 'Bún mắm': 'Bun mam', 'Bún riêu': 'Bun rieu',
            'Bún thịt nướng': 'Bun thit nuong', 'Cá kho tộ': 'Ca kho to', 'Canh chua': 'Canh chua',
            'Cao lầu': 'Cao lau', 'Cháo lòng': 'Chao long', 'Cơm tấm': 'Com tam',
            'Gỏi cuốn': 'Goi cuon', 'Hủ tiếu': 'Hu tieu', 'Mì Quảng': 'Mi quang',
            'Nem chua': 'Nem chua', 'Phở': 'Pho', 'Xôi xéo': 'Xoi xéo'
        }

        for dish_name in all_dishes:
            result = self.calculate_dish_calories(dish_name)
            if not result:
                continue
                
            total_weight = sum(ing['grams'] for ing in result['ingredients'])
            
            if total_weight > 0:
                kcal_100g = round((result['per_serving_kcal'] / total_weight) * 100, 1)
                protein_100g = round((result['per_serving_protein_g'] / total_weight) * 100, 1)
                carb_100g = round((result['per_serving_carb_g'] / total_weight) * 100, 1)
                fat_100g = round((result['per_serving_fat_g'] / total_weight) * 100, 1)
                
                class_name = mapping_class.get(dish_name, dish_name)
                
                categories_records.append({
                    'class_name': class_name,
                    'food_name_vi': dish_name,
                    'kcal_per_100g': kcal_100g,
                    'protein_g': protein_100g,
                    'carb_g': carb_100g,
                    'fat_g': fat_100g,
                    'total_weight_suat_g': round(total_weight, 1),
                    'total_suat_kcal': round(result['per_serving_kcal'], 1),
                    'source': 'NIN_Vietnam_Calculated'
                })

        df_result = pd.DataFrame(categories_records)
        
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        df_categories = df_result[['class_name', 'food_name_vi', 'kcal_per_100g', 'protein_g', 'carb_g', 'fat_g', 'source']]
        df_categories.to_csv(data_dir / 'categories.csv', index=False, encoding='utf-8-sig')
        
        df_calories = df_result[['class_name', 'food_name_vi', 'total_weight_suat_g', 'total_suat_kcal', 'source']]
        df_calories.to_csv(data_dir / 'calories.csv', index=False, encoding='utf-8-sig')
        
        print(f"🎉 Xuất file thành công tại thư mục: {data_dir.resolve()}")
        print("-> Đã sinh ra 2 file: 'categories.csv' và 'calories.csv' mới tinh!")


if __name__ == "__main__":
    # Test thử nghiệm thực tế liên thông hệ thống
    calc = RecipeCalculator()
    
    print("\n" + "🍲"*35)
    print("TEST 1: Calculate 2 servings of Pho")
    calc.print_dish_nutrition("Phở", num_servings=2)
    
    print("\n" + "📊"*35)
    print("TEST 2: Detailed Breakdown of Banh Mi")
    df = calc.get_detailed_breakdown("Bánh mì")
    if df is not None:
        print(df.to_string(index=False))
    else:
        print("❌ Không tìm thấy thông tin món ăn.")

    # === THAY ĐỔI QUAN TRỌNG Ở ĐÂY: GỌI HÀM ĐỂ THỰC THI XUẤT FILE ===
    print("\n" + "⏳"*35)
    print("Đang chạy tiến trình quét hệ thống để tạo file...")
    calc.export_all_to_csv()