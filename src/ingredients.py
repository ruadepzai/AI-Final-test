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
            csv_path: Đường dẫn tới file dữ liệu nguyên liệu (ví dụ: data/ingredients_nutrition.csv)
                    Nếu None, sẽ tự động tìm trong thư mục data/
        """
        if csv_path is None:
            # Tìm file từ vị trí thư mục của dự án
            workspace_root = Path(__file__).parent.parent
            # Bạn có thể đổi tên file thành 'data.csv' hoặc 'ingredients_nutrition.csv' cho đúng với máy bạn
            csv_path = workspace_root / 'data' / 'ingredients_nutrition.csv'
        
        self.csv_path = Path(csv_path)
        self.df = self._load_ingredients()
        self.ingredients_dict = self._create_dict()
    
    def _load_ingredients(self) -> pd.DataFrame:
        """Load ingredients từ CSV"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu CSV tại: {self.csv_path}\nHãy kiểm tra lại tên file trong hàm __init__.")
        
        df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(df)} ingredients from {self.csv_path.name}")
        return df
    
    def _create_dict(self) -> Dict:
        """Tạo dictionary cho quick lookup khớp chính xác với cột dữ liệu Việt Nam"""
        ingredients_dict = {}
        for _, row in self.df.iterrows():
            ingredient_name = row['TÊN THỨC ĂN']
            
            # Hàm bổ trợ xử lý ép kiểu số và làm sạch dấu phẩy thập phân ',' thành dấu chấm '.'
            def to_float(val):
                if pd.isna(val):
                    return 0.0
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    val = val.replace(',', '.').strip()
                    try:
                        return float(val)
                    except ValueError:
                        return 0.0
                return 0.0

            # Ánh xạ từ các cột tiếng Việt có dấu của CSV sang thuộc tính tiếng Anh của Code
            ingredients_dict[ingredient_name] = {
                'name_vi': row['TÊN THỨC ĂN'],
                'kcal_per_100g': to_float(row['Calories (kcal)']),
                'protein_g': to_float(row['Protein (g)']),
                'carb_g': to_float(row['Carbonhydrates (g)']),
                'fat_g': to_float(row['Fat (g)']),
                'unit': 'g',          # Điền mặc định vì file gốc không có cột đơn vị
                'category': row['Loại'],
                'notes': '-'          # Điền mặc định vì file gốc không có cột ghi chú
            }
        return ingredients_dict
    
    def get_ingredient(self, ingredient_name: str) -> Optional[Dict]:
        """
        Lấy thông tin nguyên liệu theo tên tiếng Việt (ví dụ: "Gạo tẻ", "Thịt bò")
        
        Args:
            ingredient_name: Tên nguyên liệu cần tìm
        
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
        
        # Tính toán tỷ lệ hiệu chỉnh (per 100g -> per grams)
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
        Tìm nguyên liệu theo danh mục (Dựa vào cột 'Loại')
        
        Args:
            category: Tên danh mục (Ví dụ: "THỊT VÀ SẢN PHẨM CHẾ BIẾN")
        
        Returns:
            DataFrame với các nguyên liệu trong danh mục đó
        """
        return self.df[self.df['Loại'].str.lower() == category.lower()]
    
    def get_categories(self) -> list:
        """Lấy danh sách tất cả các nhóm loại thực phẩm"""
        return sorted(self.df['Loại'].unique().tolist())
    
    def print_ingredients_summary(self):
        """In tóm tắt database nguyên liệu ra Terminal"""
        print("\n" + "="*70)
        print("📦 INGREDIENTS DATABASE SUMMARY (NIN VIETNAM)")
        print("="*70)
        
        categories = self.get_categories()
        for cat in categories:
            items = self.search_by_category(cat)
            print(f"\n{cat.upper()} ({len(items)} thực phẩm):")
            for _, item in items.iterrows():
                print(f"  • {item['TÊN THỨC ĂN']:35} - {item['Calories (kcal)']} kcal/100g")


# Global instance (lazy load)
_ingredients_db = None

def get_ingredients_db() -> IngredientsDB:
    """Lấy global ingredients database instance"""
    global _ingredients_db
    if _ingredients_db is None:
        _ingredients_db = IngredientsDB()
    return _ingredients_db


if __name__ == "__main__":
    # kịch bản chạy thử nghiệm kiểm tra tính đúng đắn của dữ liệu mới
    try:
        db = IngredientsDB()
        
        # Test 1: Lấy thông tin một nguyên liệu Việt Nam có sẵn
        print("\n1. Thử nghiệm lấy thông tin nguyên liệu:")
        gao = db.get_ingredient("Gạo tẻ")
        print(f"Gạo tẻ: {gao}")
        
        # Test 2: Tính toán dinh dưỡng theo khối lượng tùy chỉnh
        print("\n2. Thử nghiệm tính toán dinh dưỡng cho 200g Gạo tẻ:")
        nutrition = db.get_nutrition("Gạo tẻ", 200)
        print(f"Dinh dưỡng trong 200g Gạo tẻ: {nutrition}")
        
        # Test 3: Lọc danh mục theo nhóm Loại mới
        print("\n3. Thử nghiệm lọc danh mục Nhóm thịt:")
        meats = db.search_by_category("THỊT VÀ SẢN PHẨM CHẾ BIẾN")
        print(meats[['TÊN THỨC ĂN', 'Calories (kcal)', 'Protein (g)']].head())
        
        # Test 4: In toàn bộ báo cáo
        db.print_ingredients_summary()
        
    except Exception as e:
        print(f"❌ Vẫn còn lỗi phát sinh: {e}")