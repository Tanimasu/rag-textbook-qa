"""
文件名: parsingPDF.py
主要功能:
    - 使用 Docling 库将 PDF 文档转换为 Markdown 格式
    - 配置 EasyOCR 进行中英文 OCR 识别，支持 GPU 加速
    - 启用表格结构识别，提升表格内容提取质量
    - 输出转换后的 Markdown 文件，并统计文档结构信息
    - 记录初始化与转换各阶段耗时，提供性能分析报告
所属模块: PDF 处理
"""
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat
from pathlib import Path
# 确保路径完全正确
source = Path(r"D:\CodeField\Graduation_project\project\data\数据库原理及应用教程.pdf")
output_path = Path(r"D:\CodeField\Graduation_project\project\output\数据库原理及应用教程.md")
output_path.parent.mkdir(exist_ok=True)
if not source.exists():
    print(f"找不到文件: {source}")
else:
    print(f"开始处理: {source.name}")
    pipeline_options = PdfPipelineOptions()
    # 1. 开启 OCR
    pipeline_options.do_ocr = True
    # 2. 配置 EasyOCR 选项
    ocr_options = EasyOcrOptions(
        use_gpu=True,  # 使用 GPU 加速
        lang=['ch_sim', 'en']  # 支持简体中文和英文
    )
    pipeline_options.ocr_options = ocr_options
    # 3. 表格识别
    pipeline_options.do_table_structure = True
    try:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        # 转换文档
        result = converter.convert(source)
        markdown_text = result.document.export_to_markdown()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        print(f"✅ 转换成功！保存至: {output_path}")
        print(f"📊 结果字符数: {len(markdown_text)}")
    except Exception as e:
        print(f"💥 转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息