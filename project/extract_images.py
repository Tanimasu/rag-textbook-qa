"""
extract_images.py
从 PDF 中提取所有嵌入图片，保存为 PNG 文件。
运行方式：python extract_images.py
"""
from pathlib import Path
from docling.document_converter import DocumentConverter


def extract_images_from_pdf(pdf_path: str, output_dir: str):
    """
    将 PDF 中的所有图片提取并保存为 PNG。

    Args:
        pdf_path:   源 PDF 文件路径
        output_dir: 图片输出目录（不存在会自动创建）
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = DocumentConverter().convert(pdf_path)
    pictures = result.document.pictures

    if not pictures:
        print("未找到任何图片。")
        return

    for i, image in enumerate(pictures, 1):
        img_path = out / f"image_{i:03d}.png"
        img_path.write_bytes(image.data)
        print(f"已保存: {img_path}")

    print(f"\n共提取 {len(pictures)} 张图片 → {out}")


if __name__ == "__main__":
    # 修改这里指定要提取的 PDF 和输出目录
    PDF_PATH   = "./data/操作系统.pdf"
    OUTPUT_DIR = "./output/images"

    extract_images_from_pdf(PDF_PATH, OUTPUT_DIR)
