"""Count total images in NASA handbook"""
import fitz
from pathlib import Path

def count_images():
    pdf_path = "nasa_systems_engineering_handbook_0.pdf"
    
    print(f"\nCounting images in: {pdf_path}")
    print("=" * 70)
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    print(f"Total pages: {total_pages}\n")
    
    total_images = 0
    min_size = 10000  # 10KB minimum (same filter as Gemini extraction)
    
    for page_num in range(total_pages):
        page = doc[page_num]
        images = page.get_images()
        
        # Count only images above size threshold
        page_image_count = 0
        for img in images:
            xref = img[0]
            try:
                image_data = doc.extract_image(xref)
                if image_data and len(image_data.get("image", b"")) >= min_size:
                    page_image_count += 1
            except:
                pass
        
        if page_image_count > 0:
            print(f"Page {page_num + 1:3d}: {page_image_count} image(s)")
            total_images += page_image_count
    
    doc.close()
    
    print("\n" + "=" * 70)
    print(f"TOTAL IMAGES (≥10KB): {total_images}")
    print("=" * 70)
    
    return total_images

if __name__ == "__main__":
    count_images()
