import pandas as pd

# Load and preprocess the dataset
file_path = 'styles.csv'  # Replace with your dataset file path
data = pd.read_csv(file_path, quoting=1, on_bad_lines='skip')

# Handle missing values
data = data.dropna(subset=['baseColour', 'articleType', 'productDisplayName', 'gender'])

# Define clothing type to color mapping
color_groups = {
    'upperwear': ['Navy Blue', 'Blue', 'Silver', 'Black', 'Grey', 'Green', 'Purple', 'White',
                  'Beige', 'Brown', 'Pink', 'Red', 'Orange', 'Yellow', 'Maroon', 'Magenta', 'Rust'],
    'bottomwear': [     'Blue', 'Black', 'Navy Blue', 'White', 'Grey', 'Brown', 'Lavender','Pink', 'Peach', 'Purple', 'Turquoise Blue'],
    'footwear': ['Black', 'White', 'Brown', 'Purple', 'Red', 'Khaki', 'Blue', 'Grey',
        'Silver', 'Copper', 'Green', 'Lavender', 'Pink', 'Gold', 'Charcoal', 
        'Maroon', 'Tan', 'Beige', 'Multi', 'Yellow', 'Navy Blue', 
        'Turquoise Blue', 'Bronze', 'Cream', 'Olive', 'Taupe', 'Off White', 
        'Orange', 'Mushroom Brown', 'Metallic', 'Mustard', 'Teal', 'Rust', 
        'Coffee Brown', 'Nude', 'Magenta', 'Sea Green', 'Peach', 
        'Fluorescent Green', 'Burgundy']
}

# Function to recommend outfit specifications
def recommend_outfit_specs(body_shape, hair_color, skin_tone, eye_color, gender, clothing_type):
    """
    Generate outfit specifications based on body details and clothing type.
    """
    outfit_specs = {
        'body_shape': body_shape,
        'suggested_styles': [],
        'suggested_colors': []
    }

    # Map body shapes to styles based on gender
    body_shape_styles = {
        'women': {
            'hourglass': ['Wrap dresses', 'Fitted tops', 'Pencil skirts'],
            'inverted triangle': ['A-line skirts', 'Wide-leg pants', 'Boat-neck tops'],
            'pear': ['A-line dresses', 'High-waist pants', 'V-neck blouses'],
            'apple': ['Empire waist dresses', 'Flowy tops', 'Straight-leg pants'],
            'rectangle': ['Peplum tops', 'Ruffle dresses', 'Belted coats']
        },
        'men': {
            'inverted triangle': ['Fitted shirts', 'Slim-fit pants', 'V-neck sweaters'],
            'rectangle': ['Straight-cut blazers', 'Layered outfits', 'Casual shirts'],
            'oval': ['Vertical stripes', 'Dark solid colors', 'Comfort-fit pants'],
            'trapezoid': ['Tailored suits', 'Polo shirts', 'Chinos']
        }
    }

    # Add styles for the given body shape
    outfit_specs['suggested_styles'] = body_shape_styles.get(gender.lower(), {}).get(
        body_shape.lower(), ['Comfortable fits']
    )

    # Map skin tone and hair color to colors
    skin_hair_color_map = {
    ('cool', 'black'): ['Blue', 'Grey', 'Silver', 'White', 'Navy Blue'],
    ('cool', 'dark brown'): ['Navy Blue', 'Charcoal', 'Purple', 'Steel', 'Magenta'],
    ('cool', 'medium brown'): ['Turquoise Blue', 'Lavender', 'Magenta', 'Cream', 'Peach'],
    ('cool', 'light brown'): ['Teal', 'Sea Green', 'Metallic', 'Grey Melange'],
    ('neutral', 'black'): ['Burgundy', 'White', 'Gold', 'Off White', 'Copper'],
    ('neutral', 'dark brown'): ['Maroon', 'Beige', 'Tan', 'Cream', 'Khaki'],
    ('neutral', 'medium brown'): ['Peach', 'Cream', 'Olive', 'Rust', 'Skin'],
    ('neutral', 'light brown'): ['Taupe', 'Coffee Brown', 'Mushroom Brown', 'Charcoal', 'Brown'],
    ('warm', 'black'): ['Red', 'Yellow', 'Rust', 'Orange', 'Gold'],
    ('warm', 'dark brown'): ['Orange', 'Mustard', 'Brown', 'Burgundy', 'Tan'],
    ('warm', 'medium brown'): ['Khaki', 'Coffee Brown', 'Taupe', 'Maroon', 'Cream'],
    ('warm', 'light brown'): ['Beige', 'Peach', 'Pink', 'Lime Green', 'Rust'],
    ('olive', 'black'): ['Teal', 'Sea Green', 'Skin', 'Grey', 'Mushroom Brown'],
    ('olive', 'dark brown'): ['Rose', 'Lime Green', 'Pink', 'Turquoise Blue', 'Copper'],
    ('olive', 'medium brown'): ['Mushroom Brown', 'Fluorescent Green', 'Copper', 'Olive', 'Bronze'],
    ('olive', 'light brown'): ['Cream', 'Taupe', 'Peach', 'Charcoal', 'Metallic'],
    ('cool', 'auburn'): ['Lavender', 'Purple', 'Navy Blue', 'Grey', 'Magenta'],
    ('neutral', 'auburn'): ['Burgundy', 'Beige', 'Rust', 'Peach', 'Tan'],
    ('warm', 'auburn'): ['Red', 'Orange', 'Mustard', 'Gold', 'Brown'],
    ('olive', 'auburn'): ['Olive', 'Taupe', 'Skin', 'Sea Green', 'Lime Green'],
    ('cool', 'blonde'): ['Silver', 'Lavender', 'Blue', 'Turquoise Blue', 'Pink'],
    ('neutral', 'blonde'): ['White', 'Gold', 'Beige', 'Tan', 'Cream'],
    ('warm', 'blonde'): ['Yellow', 'Peach', 'Rust', 'Orange', 'Mustard'],
    ('olive', 'blonde'): ['Sea Green', 'Olive', 'Lime Green', 'Rose', 'Taupe'],
    ('cool', 'gray'): ['Charcoal', 'Steel', 'Silver', 'Blue', 'Grey'],
    ('neutral', 'gray'): ['White', 'Cream', 'Beige', 'Khaki', 'Copper'],
    ('warm', 'gray'): ['Rust', 'Gold', 'Orange', 'Yellow', 'Red'],
    ('olive', 'gray'): ['Teal', 'Skin', 'Mushroom Brown', 'Bronze', 'Sea Green'],
    }

    # Default colors if no match is found
    default_colors = ['Black', 'White', 'Grey']

    outfit_specs['suggested_colors'] = skin_hair_color_map.get(
        (skin_tone.lower(), hair_color.lower()), default_colors
    )

    # Use clothing type to refine colors
    clothing_type_colors = color_groups.get(clothing_type.lower(), default_colors)
    outfit_specs['suggested_colors'] = list(set(outfit_specs['suggested_colors']) & set(clothing_type_colors))

    # Add accent colors based on eye color
    eye_color_accent_map = {
        'blue': ['Navy Blue', 'Silver', 'White'],
        'green': ['Olive', 'Sea Green', 'Cream'],
        'hazel': ['Taupe', 'Brown', 'Beige'],
        'black': ['Charcoal', 'Black', 'Grey'],
        'brown': ['Burgundy', 'Maroon', 'Tan'],
        'gray': ['Steel', 'Grey Melange', 'Metallic']
    }
    outfit_specs['suggested_colors'].extend(eye_color_accent_map.get(eye_color.lower(), []))

    return outfit_specs

# Function to find matching outfits in the dataset
def find_matching_outfits(outfit_specs, selected_article_type, gender, clothing_type, num_recommendations=5, use_keyword_matches=True):
    """
    Find relevant outfits based on specifications, articleType, and gender, with optional keyword-based matching.
    """
    # Filter dataset by gender
    gender_filtered = data[data['gender'].str.strip().str.lower() == gender.strip().lower()]
    
    # Filter dataset by base color and articleType
    matching_items = gender_filtered[
        (gender_filtered['baseColour'].str.contains('|'.join(outfit_specs['suggested_colors']), case=False, na=False)) &
        (gender_filtered['articleType'].str.strip().str.lower() == selected_article_type.strip().lower())
    ]

    if use_keyword_matches:
        # Optional: Add keyword-based matching using suggested styles
        keyword_matches = matching_items[
            matching_items['productDisplayName'].str.contains('|'.join(outfit_specs['suggested_styles']), case=False, na=False)
        ]
        # Combine matches while avoiding duplicates
        all_matches = pd.concat([matching_items, keyword_matches]).drop_duplicates()
    else:
        all_matches = matching_items

    # If matches are found, return top recommendations with specific columns
    if not all_matches.empty:
        recommendations = all_matches.sample(min(num_recommendations, len(all_matches)))
        return recommendations[['id', 'usage', 'productDisplayName']]
    else:
        # If no match is found, suggest default colors for the clothing type
        default_colors = color_groups.get(clothing_type.lower(), ['Black', 'White', 'Grey'])
        fallback_items = gender_filtered[
            (gender_filtered['baseColour'].str.contains('|'.join(default_colors), case=False, na=False)) &
            (gender_filtered['articleType'].str.strip().str.lower() == selected_article_type.strip().lower())
        ]
        if not fallback_items.empty:
            fallback_recommendations = fallback_items.sample(min(num_recommendations, len(fallback_items)))
            return fallback_recommendations[['id', 'usage', 'productDisplayName']]
        
        return "No matching outfits found for the selected category, gender, and preferences."

# Example user input
body_details = {
    'body_shape': 'oval',
    'hair_color': 'black',
    'skin_tone': 'olive',
    'eye_color': 'brown'
}
selected_gender = 'Men'  # Example gender input
selected_article_type = 'Jeans' # Example articleType input
selected_clothing_type = 'bottomwear'  # Example clothing type

# Generate outfit specifications
outfit_specs = recommend_outfit_specs(**body_details, gender=selected_gender, clothing_type=selected_clothing_type)
print("Generated Outfit Specifications:")
print(outfit_specs)

# Find and display matching outfits from the dataset
recommended_outfits = find_matching_outfits(outfit_specs, selected_article_type, selected_gender, selected_clothing_type)
print("\nRecommended Outfits:")
print(recommended_outfits)


# Filter the dataset for shoes and get unique colors
shoe_colors = data[data['masterCategory'].str.strip().str.lower() == 'footwear']['articleType'].unique()

print("Unique colors for shoes in the dataset:")
print(shoe_colors)
