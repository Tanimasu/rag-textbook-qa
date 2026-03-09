"""
test_pdf_parser.py
测试 Docling PDF 解析器：将 PDF 转换为 Markdown，输出内容统计和预览。
运行方式：python test_pdf_parser.py
"""
import os
import time
import warnings
from pathlib import Path

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
warnings.filterwarnings('ignore')

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions


def test_pdf_extraction(pdf_path: str, max_pages: int = None, save_output: bool = True) -> str | None:
    """
    将 PDF 转换为 Markdown，并打印内容统计信息。

    Args:
        pdf_path:    PDF 文件路径
        max_pages:   最多转换的页数（None 表示全部）
        save_output: 是否将结果保存到 output/ 目录

    Returns:
        转换后的 Markdown 字符串，失败返回 None
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"文件不存在: {pdf_path}")
        return None

    print("=" * 60)
    print(f"PDF 文件: {pdf_file.name}")
    print(f"文件大小: {pdf_file.stat().st_size / 1024 / 1024:.2f} MB")
    if max_pages:
        print(f"限制页数: {max_pages} 页")
    print("=" * 60)

    # 配置 OCR（支持中英文，GPU 加速）
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.ocr_options = EasyOcrOptions(use_gpu=True, lang=['ch_sim', 'en'])

    try:
        # 初始化转换器
        print("初始化转换器（首次运行会下载模型，约 1-2 分钟）...")
        t0 = time.time()
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        init_time = time.time() - t0
        print(f"初始化完成（{init_time:.1f}s）")

        # 执行转换
        print("开始转换...")
        t1 = time.time()
        kwargs = {"max_num_pages": max_pages} if max_pages else {}
        result = converter.convert(str(pdf_file), **kwargs)
        convert_time = time.time() - t1
        print(f"转换完成（{convert_time:.1f}s）")

        # 统计文档元素
        stats = {}
        total_chars = 0
        for item, _level in result.document.iterate_items():
            label = getattr(item, 'label', 'unknown')
            stats[label] = stats.get(label, 0) + 1
            if hasattr(item, 'text') and item.text:
                total_chars += len(item.text)

        print("\n内容统计:")
        for label, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}")
        print(f"  总字符数: {total_chars:,}")

        # 导出 Markdown
        markdown = result.document.export_to_markdown()
        print(f"\n内容预览（前 500 字符）:")
        print("-" * 60)
        print(markdown[:500])
        print("-" * 60)
        print(f"\n耗时: 初始化 {init_time:.1f}s + 转换 {convert_time:.1f}s = {init_time + convert_time:.1f}s")

        # 保存结果
        if save_output:
            out_path = Path("output") / (pdf_file.stem + ".md")
            out_path.parent.mkdir(exist_ok=True)
            out_path.write_text(markdown, encoding="utf-8")
            print(f"已保存: {out_path}")

        return markdown

    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 修改这里指定要测试的 PDF
    PDF_PATH = "data/操作系统.pdf"
    MAX_PAGES = 5  # 测试时只转换前5页，改为 None 则处理全部

    test_pdf_extraction(PDF_PATH, max_pages=MAX_PAGES)
