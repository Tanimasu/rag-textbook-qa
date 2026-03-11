"""
文件名: parsingPDF_mineru.py
主要功能:
    - 使用 MinerU 将指定 PDF 转换为 Markdown 格式
    - parse_method="auto" 自动判断每页是否需要 OCR（数字版直接提取，扫描版走 OCR）
    - 支持中英文混合内容，表格和公式识别
    - 对长 PDF 自动分段处理（每段 CHUNK_PAGES 页），避免公式识别阶段 CUDA OOM
    - 输出文件名加 _mineru 后缀，与 Docling 版本区分
所属模块: PDF 处理
"""

import os
import shutil
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import fitz  # pymupdf（MinerU 依赖，无需额外安装）
from mineru.cli.common import do_parse, read_fn


# ===================== 配置路径（每次修改这里）=====================
source = Path(r"D:\CodeField\Graduation_project\project\data\数据库原理及应用教程.pdf")
output_path = Path(r"D:\CodeField\Graduation_project\project\output\数据库原理及应用教程_mineru.md")
# =================================================================

# 每段最多处理的页数
CHUNK_PAGES = 100


def parse_chunk(pdf_bytes: bytes, chunk_name: str, tmp_dir: Path, backend: str) -> str:
    """处理单个分段，返回 Markdown 文本。"""
    do_parse(
        output_dir=str(tmp_dir),
        pdf_file_names=[chunk_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=["ch"],
        backend=backend,
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        f_dump_md=True,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=False,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
    )

    generated_md = tmp_dir / chunk_name / "auto" / f"{chunk_name}.md"
    if not generated_md.exists():
        candidates = list((tmp_dir / chunk_name).rglob("*.md"))
        if not candidates:
            raise FileNotFoundError(f"[错误] 分段 {chunk_name} 未生成 Markdown")
        generated_md = candidates[0]

    return generated_md.read_text(encoding="utf-8")


def parse_pdf(pdf_path: Path, dest: Path, backend: str = "pipeline"):
    tmp_dir = dest.parent / "_mineru_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_name = pdf_path.stem

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"开始处理: {pdf_path.name}  ({total_pages} 页)")
    print(f"backend={backend}, parse_method=auto, lang=ch, formula_enable=True")

    if total_pages <= CHUNK_PAGES:
        # 页数不多，直接处理
        print("页数较少，整本处理...")
        pdf_bytes = read_fn(str(pdf_path))
        md_text = parse_chunk(pdf_bytes, file_name, tmp_dir, backend)
    else:
        # 分段处理
        chunks_needed = (total_pages + CHUNK_PAGES - 1) // CHUNK_PAGES
        print(f"页数较多，分 {chunks_needed} 段处理（每段 {CHUNK_PAGES} 页）...")
        md_parts = []

        for i in range(chunks_needed):
            start = i * CHUNK_PAGES
            end = min(start + CHUNK_PAGES - 1, total_pages - 1)
            chunk_name = f"{file_name}_part{i + 1:02d}"
            print(f"\n  [{i + 1}/{chunks_needed}] 处理第 {start + 1}-{end + 1} 页 → {chunk_name}")

            # 提取分段 PDF 到内存
            sub_doc = fitz.open()
            sub_doc.insert_pdf(doc, from_page=start, to_page=end)
            chunk_bytes = sub_doc.tobytes()
            sub_doc.close()

            md_text = parse_chunk(chunk_bytes, chunk_name, tmp_dir, backend)
            md_parts.append(md_text)
            print(f"  段落字符数: {len(md_text):,}")

            # 清理该段临时文件，释放磁盘空间
            chunk_tmp = tmp_dir / chunk_name
            if chunk_tmp.exists():
                shutil.rmtree(chunk_tmp)

        md_text = "\n\n".join(md_parts)

    doc.close()

    dest.parent.mkdir(exist_ok=True)
    dest.write_text(md_text, encoding="utf-8")
    print(f"\n转换成功！保存至: {dest}")
    print(f"结果字符数: {len(md_text):,}")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    if not source.exists():
        print(f"找不到文件: {source}")
    else:
        print(f"输入: {source.name}  ({source.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"输出: {output_path}")
        try:
            parse_pdf(source, output_path)
        except Exception as e:
            print(f"转换过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
