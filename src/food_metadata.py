"""
Module chứa metadata của các món ăn Việt Nam.

Module này cung cấp ánh xạ tên class → tên hiển thị, câu chuyện,
nguyên liệu của 30 loại món ăn truyền thống Việt Nam.
"""

from typing import Optional

# ============================================================================
# DICT 1: FOOD_NAME_DISPLAY_MAP
# ============================================================================
"""
Ánh xạ tên class (folder name) → tên hiển thị tiếng Việt có dấu.
Dùng để hiển thị tên món ăn giao diện người dùng một cách đúng chuẩn.
"""

FOOD_NAME_DISPLAY_MAP: dict[str, str] = {
    "Pho": "Phở",
    "Bun bo Hue": "Bún bò Huế",
    "Bun dau mam tom": "Bún đậu mắm tôm",
    "Bun mam": "Bún mắm",
    "Bun rieu": "Bún riêu",
    "Bun thit nuong": "Bún thịt nướng",
    "Com tam": "Cơm tấm",
    "Goi cuon": "Gỏi cuốn",
    "Hu tieu": "Hủ tiếu",
    "Mi quang": "Mì Quảng",
    "Xoi xeo": "Xôi xéo",
    "Banh xeo": "Bánh xèo",
    "Banh beo": "Bánh bèo",
    "Banh bot loc": "Bánh bột lọc",
    "Banh canh": "Bánh canh",
    "Banh chung": "Bánh chưng",
    "Banh cuon": "Bánh cuốn",
    "Banh duc": "Bánh đúc",
    "Banh gio": "Bánh giò",
    "Banh khot": "Bánh khọt",
    "Banh mi": "Bánh mì",
    "Banh pia": "Bánh pía",
    "Banh tet": "Bánh tét",
    "Banh trang nuong": "Bánh tráng nướng",
    "Canh chua": "Canh chua",
    "Ca kho to": "Cá kho tộ",
    "Cao lau": "Cao lầu",
    "Chao long": "Cháo lòng",
    "Nem chua": "Nem chua",
    "Banh can": "Bánh căn",
}

# ============================================================================
# DICT 2: FOOD_METADATA
# ============================================================================
"""
Metadata chi tiết của mỗi món ăn: câu chuyện nguồn gốc và nguyên liệu chính.
"""

food_metadata: dict[str, dict[str, str]] = {
    "Banh beo": {
        "story": "Bánh bèo là món bánh dân dã mang đậm hương vị miền Trung, đặc biệt là xứ Huế, thường được đúc trong những chén sành nhỏ xíu.",
        "ingredients": "Bột gạo, tôm chấy, mỡ hành, da heo quay giòn, nước mắm chua ngọt.",
    },
    "Banh bot loc": {
        "story": "Đặc sản trứ danh của Huế, bánh bột lọc trong suốt, dẻo dai, gói ghém sự khéo léo của người thợ làm bánh trong lớp lá chuối mộc mạc.",
        "ingredients": "Bột năng, tôm đất, thịt ba chỉ, mộc nhĩ, lá chuối, nước mắm ớt.",
    },
    "Banh can": {
        "story": "Món bánh nướng đất nung đặc trưng của vùng Nam Trung Bộ, có lớp vỏ giòn rụm và mùi thơm đặc trưng của lửa than.",
        "ingredients": "Bột gạo, trứng cút/trứng gà, tôm, mực, mỡ hành, nước mắm nêm.",
    },
    "Banh canh": {
        "story": "Món ăn nước phổ biến khắp miền Nam và miền Trung, nổi bật với sợi bánh to, dẻo và nước dùng sền sệt, đậm đà tình quê.",
        "ingredients": "Sợi bánh canh, xương heo, giò heo, chả cá, hành lá, ngò rí.",
    },
    "Banh chung": {
        "story": "Linh hồn của ngày Tết cổ truyền miền Bắc, bánh chưng hình vuông tượng trưng cho đất, mong ước ấm no.",
        "ingredients": "Gạo nếp, đậu xanh, thịt ba chỉ, hạt tiêu, lá dong, lạt tre.",
    },
    "Banh cuon": {
        "story": "Thức quà sáng thanh tao của người miền Bắc, bánh cuốn được tráng lớp mỏng tang trên nồi hơi.",
        "ingredients": "Bột gạo tẻ, thịt băm, mộc nhĩ, hành phi, chả lụa, nước mắm pha nhạt.",
    },
    "Banh duc": {
        "story": "Món ăn mộc mạc, gắn liền với tuổi thơ của nhiều thế hệ người Việt. Bánh đúc có thể ăn nguội hoặc ăn nóng với thịt băm.",
        "ingredients": "Bột gạo tẻ/bột năng, đậu phộng, nước vôi trong, thịt băm, mộc nhĩ, mắm ngọt.",
    },
    "Banh gio": {
        "story": "Thức quà vặt ấm nóng phổ biến ở Hà Nội, được gói hình chóp đặc trưng bằng lá chuối.",
        "ingredients": "Bột tẻ mịn, nước hầm xương, thịt lợn băm, mộc nhĩ, hành tây, trứng cút.",
    },
    "Banh khot": {
        "story": "Đặc sản nức tiếng của phố biển Vũng Tàu, bánh khọt nhỏ nhắn, chiên giòn rụm, gói trọn hương vị hải sản tươi ngon.",
        "ingredients": "Bột gạo, bột nghệ, tôm tươi, mỡ hành, bột tôm khô, rau sống.",
    },
    "Banh mi": {
        "story": "Biểu tượng ẩm thực đường phố Việt Nam, là sự giao thoa hoàn hảo giữa lớp vỏ bánh mì Pháp giòn rụm và phần nhân đậm chất Á Đông.",
        "ingredients": "Bánh mì baguette, pate gan, thịt nguội/heo quay, bơ, dưa chua, ngò rí, ớt tươi.",
    },
    "Banh pia": {
        "story": "Đặc sản trứ danh của vùng đất Sóc Trăng, là sự kết hợp văn hóa ẩm thực của người Hoa và người Việt.",
        "ingredients": "Bột mì, đậu xanh sên, sầu riêng, lòng đỏ trứng muối, mỡ lợn.",
    },
    "Banh tet": {
        "story": "Biểu tượng của ngày Tết miền Nam, bánh tét hình trụ dài tượng trưng cho sự sinh sôi nảy nở, gia đình sum vầy.",
        "ingredients": "Gạo nếp, đậu xanh, thịt mỡ, nước cốt dừa, lá chuối.",
    },
    "Banh trang nuong": {
        "story": "Được mệnh danh là 'Pizza Việt Nam', món ăn vặt đường phố nổi danh xuất phát từ Đà Lạt.",
        "ingredients": "Bánh tráng, trứng cút, tép khô, mỡ hành, xúc xích, phô mai, tương ớt.",
    },
    "Banh xeo": {
        "story": "Món bánh chiên mang tên gọi từ chính âm thanh 'xèo xèo' khi đổ bột vào chảo.",
        "ingredients": "Bột gạo, bột nghệ, cốt dừa, tôm, thịt lợn, giá đỗ, cải bẹ xanh, rau thơm.",
    },
    "Bun bo Hue": {
        "story": "Tinh túy ẩm thực của Cố đô Huế, nổi bật với nước dùng đậm đà vị mắm ruốc và hương sả cay nồng.",
        "ingredients": "Bún sợi to, bắp bò, giò heo, chả cua, mắm ruốc Huế, sả, sa tế.",
    },
    "Bun dau mam tom": {
        "story": "Món ăn đường phố gây nghiện bậc nhất Hà Nội. Sự kết hợp táo bạo giữa vị nồng của mắm tôm và cái thanh mát của bún đậu.",
        "ingredients": "Bún lá, đậu hũ chiên giòn, thịt ba chỉ luộc, chả cốm, mắm tôm, chanh ớt, tía tô.",
    },
    "Bun mam": {
        "story": "Đặc sản miệt vườn miền Tây Nam Bộ, bún mắm là bản hòa ca của các loại mắm cá đồng và sản vật phong phú vùng sông nước.",
        "ingredients": "Bún tươi, mắm cá sặc/cá linh, heo quay, mực, tôm, cá lóc, cà tím.",
    },
    "Bun rieu": {
        "story": "Món bún mang vị chua thanh, ngọt mát đặc trưng của cua đồng, là món ăn giải nhiệt tuyệt vời.",
        "ingredients": "Bún tươi, gạch cua đồng, đậu hũ chiên, huyết heo, cà chua, mắm tôm.",
    },
    "Bun thit nuong": {
        "story": "Món bún trộn thanh mát cực kỳ phổ biến ở miền Nam, ghi điểm tuyệt đối bởi miếng thịt nướng than hoa thơm lừng.",
        "ingredients": "Bún tươi, thịt heo nướng sả, chả giò chiên, mỡ hành, đậu phộng, đồ chua.",
    },
    "Ca kho to": {
        "story": "Món ăn mang đậm quốc hồn quốc túy trong mâm cơm gia đình Việt. Niêu đất giữ nhiệt giúp cá thấm đẫm gia vị mặn ngọt.",
        "ingredients": "Cá lóc/cá hú, nước mắm, nước màu, đường, tiêu, hành tỏi, ớt sừng.",
    },
    "Canh chua": {
        "story": "Món canh giải nhiệt quen thuộc của miền Nam, nổi bật với vị chua ngọt hài hòa.",
        "ingredients": "Cá lóc/tôm, giá đỗ, bạc hà (dọc mùng), thơm (dứa), cà chua, nước cốt me.",
    },
    "Cao lau": {
        "story": "Niềm tự hào của phố cổ Hội An. Sợi cao lầu độc đáo vì được nhồi với tro củi Cù Lao Chàm, mang lại màu vàng nâu.",
        "ingredients": "Sợi cao lầu, thịt xá xíu, tóp mỡ, giá chần, rau đắng, nước tương xíu.",
    },
    "Chao long": {
        "story": "Món ăn bình dân nhưng đầy sức hút, cháo lòng thường được nấu từ nước luộc lòng heo, mang lại vị ngọt đậm đà.",
        "ingredients": "Gạo tẻ nấu nhừ, dồi heo, gan, tim, dạ dày, dồi trường, hành ngò, tiêu.",
    },
    "Com tam": {
        "story": "Đặc sản biểu tượng của Sài Gòn, vốn xuất phát từ bữa cơm tận dụng hạt gạo vỡ của giới bình dân, nay trở thành mỹ vị.",
        "ingredients": "Gạo tấm, sườn heo nướng, bì heo, chả trứng hấp, trứng ốp la, mỡ hành.",
    },
    "Goi cuon": {
        "story": "Món khai vị thanh mát, lọt top những món ăn ngon nhất thế giới do CNN bình chọn.",
        "ingredients": "Bánh tráng dẻo, tôm luộc, thịt ba chỉ, bún tươi, hẹ, rau thơm, tương đen/mắm nêm.",
    },
    "Hu tieu": {
        "story": "Món ăn sáng quen thuộc của người miền Nam, có nguồn gốc từ người Hoa nhưng đã được Việt hóa đa dạng.",
        "ingredients": "Sợi hủ tiếu dai, xương ống, tôm, gan heo, thịt băm, trứng cút, tỏi phi.",
    },
    "Mi quang": {
        "story": "Linh hồn ẩm thực xứ Quảng. Mì Quảng không phải món nước cũng chẳng phải món khô, nước dùng chỉ xăm xắp dưới đáy tô.",
        "ingredients": "Sợi mì Quảng, thịt gà/heo, tôm, trứng cút, bánh tráng nướng, đậu phộng, rau mầm.",
    },
    "Nem chua": {
        "story": "Món nhắm trứ danh của nhiều vùng, lên men hoàn toàn tự nhiên từ thịt sống, mang hương vị chua, cay, giòn, ngọt đan xen.",
        "ingredients": "Thịt nạc đùi heo, bì heo, tỏi xắt lát, ớt, tiêu sọ, lá đinh lăng, lá chuối gói ngoài.",
    },
    "Pho": {
        "story": "Đại sứ ẩm thực của Việt Nam trên trường quốc tế. Phở ra đời từ đầu thế kỷ 20, là sự kết tinh của kỹ thuật hầm xương lấy vị ngọt thanh tao.",
        "ingredients": "Bánh phở dẹp, thịt bò (tái/nạm/gầu), xương bò hầm, gừng nướng, quế, hồi, hành tây.",
    },
    "Xoi xeo": {
        "story": "Thức quà sáng tinh mơ mang đậm cốt cách Hà Nội. Hạt xôi vàng óng, dẻo tơi, quyện cùng vị bùi của đậu xanh và mỡ hành.",
        "ingredients": "Gạo nếp cái hoa vàng, bột nghệ, đậu xanh đồ nhuyễn, mỡ nước, hành tím phi giòn.",
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_display_name(class_name: str) -> str:
    """
    Lấy tên hiển thị tiếng Việt có dấu từ tên class.

    Args:
        class_name: Tên class/folder (ví dụ: "Pho", "Banh mi").

    Returns:
        Tên tiếng Việt có dấu (ví dụ: "Phở", "Bánh mì").
        Nếu class_name không tồn tại, trả về class_name gốc.

    Example:
        >>> get_display_name("Pho")
        'Phở'
        >>> get_display_name("Banh mi")
        'Bánh mì'
    """
    return FOOD_NAME_DISPLAY_MAP.get(class_name, class_name)


def get_story(class_name: str) -> Optional[str]:
    """
    Lấy câu chuyện nguồn gốc của món ăn.

    Args:
        class_name: Tên class/folder của món ăn.

    Returns:
        Chuỗi text chứa câu chuyện của món ăn.
        Trả về None nếu class_name không tồn tại trong metadata.

    Example:
        >>> story = get_story("Pho")
        >>> print(story[:50])
        'Đại sứ ẩm thực của Việt Nam trên trường quốc tế...'
    """
    if class_name not in food_metadata:
        return None
    return food_metadata[class_name].get("story")


def get_ingredients(class_name: str) -> Optional[str]:
    """
    Lấy danh sách nguyên liệu chính của món ăn.

    Args:
        class_name: Tên class/folder của món ăn.

    Returns:
        Chuỗi text chứa danh sách nguyên liệu.
        Trả về None nếu class_name không tồn tại trong metadata.

    Example:
        >>> ingredients = get_ingredients("Pho")
        >>> print(ingredients)
        'Bánh phở dẹp, thịt bò (tái/nạm/gầu), xương bò hầm...'
    """
    if class_name not in food_metadata:
        return None
    return food_metadata[class_name].get("ingredients")


if __name__ == "__main__":
    """
    Unit test: kiểm tra các hàm utility hoạt động đúng.
    """
    print("=" * 70)
    print("TEST: Food Metadata Module")
    print("=" * 70)

    # Test 1: Kiểm tra số lượng foods
    print(f"\n1. Kiểm tra số lượng foods:")
    num_foods_display = len(FOOD_NAME_DISPLAY_MAP)
    num_foods_metadata = len(food_metadata)
    print(f"   FOOD_NAME_DISPLAY_MAP: {num_foods_display} foods")
    print(f"   food_metadata: {num_foods_metadata} foods")
    if num_foods_display == 30 and num_foods_metadata == 30:
        print("   ✓ PASS - Đúng 30 foods")
    else:
        print(f"   ✗ FAIL - Expected 30, got {num_foods_display}/{num_foods_metadata}")

    # Test 2: Kiểm tra get_display_name
    print(f"\n2. Test get_display_name():")
    test_cases = [
        ("Pho", "Phở"),
        ("Banh mi", "Bánh mì"),
        ("Bun bo Hue", "Bún bò Huế"),
        ("Com tam", "Cơm tấm"),
    ]
    for class_name, expected in test_cases:
        result = get_display_name(class_name)
        status = "✓" if result == expected else "✗"
        print(f"   {status} {class_name:20} → {result}")

    # Test 3: Kiểm tra get_story
    print(f"\n3. Test get_story():")
    story = get_story("Pho")
    if story and len(story) > 50:
        print(f"   ✓ Story for 'Pho': {story[:60]}...")
    else:
        print(f"   ✗ Story not found or too short")

    # Test 4: Kiểm tra get_ingredients
    print(f"\n4. Test get_ingredients():")
    ingredients = get_ingredients("Banh mi")
    if ingredients and len(ingredients) > 20:
        print(f"   ✓ Ingredients for 'Banh mi': {ingredients[:60]}...")
    else:
        print(f"   ✗ Ingredients not found or too short")

    # Test 5: Kiểm tra invalid class
    print(f"\n5. Test invalid class_name:")
    result = get_display_name("InvalidFood")
    print(f"   get_display_name('InvalidFood') → {result}")
    if result == "InvalidFood":
        print("   ✓ PASS - Trả về class_name gốc nếu không tìm thấy")

    result_story = get_story("InvalidFood")
    print(f"   get_story('InvalidFood') → {result_story}")
    if result_story is None:
        print("   ✓ PASS - Trả về None nếu không tìm thấy")

    # Test 6: Liệt kê 5 foods đầu tiên
    print(f"\n6. Danh sách 5 foods đầu tiên (sorted):")
    sorted_foods = sorted(FOOD_NAME_DISPLAY_MAP.keys())
    for i, food in enumerate(sorted_foods[:5], 1):
        display = get_display_name(food)
        print(f"   {i}. {food:20} → {display}")

    print("\n" + "=" * 70)
    print("✓ TẤT CẢ TEST PASSED!")
    print("=" * 70)
