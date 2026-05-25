"""
Ingredients Database Management
Quản lý database nguyên liệu cho các món ăn Việt Nam
"""

from pathlib import Path
import pandas as pd
from typing import Dict, Optional


class IngredientsDB:
    """Database nguyên liệu với thông tin dinh dưỡng"""
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Khởi tạo database nguyên liệu
        
        Args:
            csv_path: Đường dẫn tới ingredients.csv
                    Nếu None, sẽ tìm trong data/ folder
        """
        if csv_path is None:
            # Tìm file từ workspace root
            workspace_root = Path(__file__).parent.parent
            csv_path = workspace_root / 'data' / 'ingredients.csv'
        
        self.csv_path = Path(csv_path)
        self.df = self._load_ingredients()
        self.ingredients_dict = self._create_dict()
    
    def _load_ingredients(self) -> pd.DataFrame:
        """Load ingredients từ CSV"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Ingredients CSV not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(df)} ingredients from {self.csv_path.name}")
        return df
    
    def _create_dict(self) -> Dict:
        """Tạo dictionary cho quick lookup"""
        ingredients_dict = {}
        for _, row in self.df.iterrows():
            ingredient_name = row['ingredient_name']
            ingredients_dict[ingredient_name] = {
                'name_vi': row['ingredient_name_vi'],
                'kcal_per_100g': float(row['kcal_per_100g']),
                'protein_g': float(row['protein_g']),
                'carb_g': float(row['carb_g']),
                'fat_g': float(row['fat_g']),
                'unit': row['unit'],
                'category': row['category'],
                'notes': row['notes']
            }
        return ingredients_dict
    
    def get_ingredient(self, ingredient_name: str) -> Optional[Dict]:
        """
        Lấy thông tin nguyên liệu theo tên
        
        Args:
            ingredient_name: Tên nguyên liệu (tiếng Anh)
        
        Returns:
            Dict với thông tin dinh dưỡng, hoặc None nếu không tìm thấy
        """
        return self.ingredients_dict.get(ingredient_name)
    
    def get_nutrition(self, ingredient_name: str, grams: float) -> Optional[Dict]:
        """
        Tính dinh dưỡng cho lượng nguyên liệu cụ thể
        
        Args:
            ingredient_name: Tên nguyên liệu
            grams: Lượng gam
        
        Returns:
            Dict: {kcal, protein_g, carb_g, fat_g}
        """
        ingredient = self.get_ingredient(ingredient_name)
        if ingredient is None:
            return None
        
        # Tính toán (per 100g -> per grams)
        factor = grams / 100.0
        
        return {
            'kcal': ingredient['kcal_per_100g'] * factor,
            'protein_g': ingredient['protein_g'] * factor,
            'carb_g': ingredient['carb_g'] * factor,
            'fat_g': ingredient['fat_g'] * factor,
        }
    
    def list_all_ingredients(self) -> pd.DataFrame:
        """Liệt kê tất cả nguyên liệu"""
        return self.df.copy()
    
    def search_by_category(self, category: str) -> pd.DataFrame:
        """
        Tìm nguyên liệu theo danh mục
        
        Args:
            category: Danh mục (Meat, Vegetables, Herbs, v.v...)
        
        Returns:
            DataFrame với các nguyên liệu trong category
        """
        return self.df[self.df['category'].str.lower() == category.lower()]
    
    def get_categories(self) -> list:
        """Lấy danh sách tất cả categories"""
        return sorted(self.df['category'].unique().tolist())
    
    def print_ingredients_summary(self):
        """In tóm tắt database nguyên liệu"""
        print("\n" + "="*70)
        print("📦 INGREDIENTS DATABASE SUMMARY")
        print("="*70)
        
        categories = self.get_categories()
        for cat in categories:
            items = self.search_by_category(cat)
            print(f"\n{cat.upper()} ({len(items)} items):")
            for _, item in items.iterrows():
                print(f"  • {item['ingredient_name']:20} ({item['ingredient_name_vi']:15}) - {item['kcal_per_100g']:.0f} kcal/100g")


# Global instance (lazy load)
_ingredients_db = None

def get_ingredients_db() -> IngredientsDB:
    """Lấy global ingredients database instance"""
    global _ingredients_db
    if _ingredients_db is None:
        _ingredients_db = IngredientsDB()
    return _ingredients_db


if __name__ == "__main__":
    # Test
    db = IngredientsDB()
    
    # Test 1: Get ingredient
    print("\n1. Get Ingredient Info:")
    beef = db.get_ingredient("Beef")
    print(f"Beef: {beef}")
    
    # Test 2: Calculate nutrition for specific amount
    print("\n2. Calculate Nutrition for 100g Beef:")
    nutrition = db.get_nutrition("Beef", 100)
    print(f"100g Beef: {nutrition}")
    
    # Test 3: List by category
    print("\n3. Meat Ingredients:")
    meats = db.search_by_category("Meat")
    print(meats[['ingredient_name', 'ingredient_name_vi', 'kcal_per_100g']])
    
    # Test 4: Summary
    db.print_ingredients_summary()
