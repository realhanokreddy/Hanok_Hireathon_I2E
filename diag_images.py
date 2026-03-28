import pymupdf as fitz
import re

pdf_path = r'data/nasa_handbook.pdf'
doc = fitz.open(pdf_path)
print(f'Total pages: {len(doc)}')

MIN_SIZE = 10000
MIN_DIM = 100

pages_with_images = []
for page_num in range(len(doc)):
    page = doc[page_num]
    imgs = page.get_images(full=True)
    if imgs:
        pages_with_images.append((page_num+1, imgs))

print(f'Pages with any images: {len(pages_with_images)}')

# Show first 10 pages with images
for page_1idx, imgs in pages_with_images[:10]:
    print(f'\nPage {page_1idx}: {len(imgs)} image(s)')
    for img in imgs:
        xref = img[0]
        base = doc.extract_image(xref)
        sz = len(base['image'])
        w = base.get('width', 0)
        h = base.get('height', 0)
        passes_size = sz >= MIN_SIZE and w >= MIN_DIM and h >= MIN_DIM
        print(f'  xref={xref} size={sz} w={w} h={h} passes_filter={passes_size}')

        if passes_size:
            page = doc[page_1idx - 1]
            # Check FIGURE text search
            hits = page.search_for('FIGURE') + page.search_for('Figure') + page.search_for('Fig.')
            print(f'  FIGURE hits on page: {len(hits)}')

            # Try get_image_bbox
            try:
                img_rect = page.get_image_bbox(img)
                print(f'  img_rect={img_rect}')
            except Exception as e:
                print(f'  get_image_bbox failed: {e}')
                img_rects = page.get_image_rects(xref)
                print(f'  get_image_rects: {img_rects}')
                if img_rects:
                    img_rect = img_rects[0]
                else:
                    img_rect = None

            if img_rect and hits:
                for fig_rect in hits[:2]:
                    dist_below = fig_rect.y0 - img_rect.y1 if fig_rect.y0 >= img_rect.y1 else None
                    dist_above = img_rect.y0 - fig_rect.y1 if fig_rect.y1 <= img_rect.y0 else None
                    dist = dist_below or dist_above or 0
                    raw = page.get_textbox(fitz.Rect(fig_rect.x0, fig_rect.y0, page.rect.x1-30, fig_rect.y1+120)).strip()
                    raw_clean = ' '.join(raw.split())
                    print(f'  fig_rect={fig_rect} dist={dist:.1f} raw={repr(raw_clean[:100])}')

doc.close()
