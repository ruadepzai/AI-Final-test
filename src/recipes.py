"""
Recipes Database Management
Quản lý công thức (recipe) các món ăn Việt Nam
"""
import sys
from pathlib import Path

# Tự động tìm đường dẫn thư mục gốc (AI-Final-test) và nạp vào hệ thống
workspace_root = str(Path(__file__).parent.parent)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
from pathlib import Path
import pandas as pd
from typing import Dict, Optional, List
from src.ingredients import get_ingredients_db


class RecipesDB:
    """Database công thức các món ăn"""
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Khởi tạo database công thức
        
        Args:
            csv_path: Đường dẫn tới recipes.csv
                    Nếu None, sẽ tìm trong data/ folder
        """
        if csv_path is None:
            workspace_root = Path(__file__).parent.parent
            csv_path = workspace_root / 'data' / 'recipes.csv'
        
        self.csv_path = Path(csv_path)
        self.df = self._load_recipes()
        self.recipes_dict = self._create_dict()
        self.ingredients_db = get_ingredients_db()
    
    def _load_recipes(self) -> pd.DataFrame:
        """Load recipes từ CSV"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Recipes CSV not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded recipes from {self.csv_path.name}")
        return df
    
    def _create_dict(self) -> Dict:
        """Tạo dictionary công thức theo tên_món"""
        recipes_dict = {}
        # CHỈ ĐỔI TÊN CỘT: 'dish_name' -> 'tên_món'
        for dish_name in self.df['tên_món'].unique():
            dish_recipes = self.df[self.df['tên_món'] == dish_name]
            
            ingredients_list = []
            for _, row in dish_recipes.iterrows():
                # CHỈ ĐỔI TÊN CỘT Ở VẾ PHẢI: Khớp với header thực tế của file CSV
                ingredients_list.append({
                    'ingredient_name': row['nguyên_liệu'],
                    'grams': float(row['grams']),
                    'notes': row['ghi_chú'] if pd.notna(row['ghi_chú']) else '-'
                })
            
            recipes_dict[dish_name] = ingredients_list
        
        return recipes_dict
    
    def get_recipe(self, dish_name: str) -> Optional[List[Dict]]:
        """
        Lấy công thức theo tên dish
        
        Args:
            dish_name: Tên món ăn
        
        Returns:
            List[Dict] với các ingredient: [
                {'ingredient_name': '...', 'grams': 100, 'notes': '...'},
                ...
            ]
        """
        return self.recipes_dict.get(dish_name)
    
    def get_all_dishes(self) -> List[str]:
        """Lấy danh sách tất cả dishes"""
        return sorted(self.recipes_dict.keys())
    
    def get_ingredients_for_dish(self, dish_name: str) -> Optional[pd.DataFrame]:
        """Lấy ingredients của dish dưới dạng DataFrame"""
        recipe = self.get_recipe(dish_name)
        if recipe is None:
            return None
        
        # CHỈ ĐỔI TÊN CỘT: 'dish_name' -> 'tên_món'
        return self.df[self.df['tên_món'] == dish_name].copy()
    
    def print_recipe(self, dish_name: str):
        """In công thức dễ đọc"""
        recipe = self.get_recipe(dish_name)
        if recipe is None:
            print(f"❌ Dish '{dish_name}' not found!")
            return
        
        print(f"\n{'='*70}")
        print(f"🍲 {dish_name.upper()} - RECIPE")
        print(f"{'='*70}")
        
        total_grams = 0
        for ingredient in recipe:
            ing_name = ingredient['ingredient_name']
            grams = ingredient['grams']
            notes = ingredient['notes']
            
            ing_db = self.ingredients_db.get_ingredient(ing_name)
            ing_name_vi = ing_db['name_vi'] if ing_db else ing_name
            
            print(f"  • {ing_name:20} ({ing_name_vi:15}) {grams:6.1f}g  [{notes}]")
            total_grams += grams
        
        print(f"{'─'*70}")
        print(f"Total weight: {total_grams:.0f}g (approx {total_grams/100:.1f} cups/bowls)")
        print(f"{'='*70}\n")
    
    def print_all_dishes(self):
        """In danh sách tất cả dishes"""
        dishes = self.get_all_dishes()
        print(f"\n{'='*70}")
        print(f"📜 AVAILABLE DISHES ({len(dishes)} dishes)")
        print(f"{'='*70}")
        
        for i, dish in enumerate(dishes, 1):
            recipe = self.get_recipe(dish)
            num_ingredients = len(recipe) if recipe else 0
            print(f"{i:2d}. {dish:20} ({num_ingredients} ingredients)")


# Global instance
_recipes_db = None

def get_recipes_db() -> RecipesDB:
    """Lấy global recipes database instance"""
    global _recipes_db
    if _recipes_db is None:
        _recipes_db = RecipesDB()
    return _recipes_db


if __name__ == "__main__":
    # Test nội bộ bằng tên món tiếng Việt có dấu khớp với file CSV mới
    db = RecipesDB()
    
    # Test 1: List all dishes
    db.print_all_dishes()
    
    # Test 2: Print specific recipe
    db.print_recipe("Phở")
    db.print_recipe("Bánh mì")