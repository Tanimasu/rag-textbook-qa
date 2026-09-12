"""Compact complete HTML tables without inventing or cutting cell values."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_TABLE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self.row: list[tuple[str, int, int]] | None = None
        self.cell: list[str] | None = None
        self.spans = (1, 1)
        self.depth = 0
        self.invalid = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.depth += 1
            self.invalid |= self.depth > 1
        elif tag == "tr":
            if self.row is not None:
                self.invalid = True
            self.row = []
        elif tag in {"td", "th"}:
            if self.cell is not None or self.row is None:
                self.invalid = True
            self.cell = []
            attributes = dict(attrs)
            try:
                self.spans = tuple(int(attributes.get(key) or 1) for key in ("rowspan", "colspan"))
                if any(span < 1 or span > 100 for span in self.spans):
                    self.invalid = True
            except ValueError:
                self.invalid = True
        elif tag == "img":
            # Images cannot be faithfully represented by stripping their tags.
            self.invalid = True
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self.depth -= 1
        elif tag in {"td", "th"}:
            if self.cell is None or self.row is None:
                self.invalid = True
                return
            self.row.append((" ".join("".join(self.cell).split()), *self.spans))
            self.cell = None
        elif tag == "tr":
            if self.row is None or self.cell is not None:
                self.invalid = True
                return
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)
        elif data.strip():
            # Captions and text outside cells need a separate representation.
            self.invalid = True

    def render(self) -> str | None:
        if self.invalid or not self.rows or self.cell is not None or self.row is not None:
            return None
        grid: dict[tuple[int, int], str] = {}
        for row_index, row in enumerate(self.rows):
            column = 0
            for value, height, width in row:
                while (row_index, column) in grid:
                    column += 1
                if row_index + height > len(self.rows):
                    return None
                for i in range(row_index, row_index + height):
                    for j in range(column, column + width):
                        if (i, j) in grid:
                            return None
                        grid[i, j] = value
                column += width
        width = max((column for _, column in grid), default=-1) + 1
        if not width:
            return None
        return "\n".join(
            "行" + str(i + 1) + ": " + " | ".join(
                grid.get((i, j), "").replace("|", "｜") for j in range(width)
            )
            for i in range(len(self.rows))
        )


def evidence_excerpt(content: str, budget: int) -> tuple[str, bool, bool]:
    """Return evidence, omission status and whether HTML was compacted.

    Table cells spanning rows or columns are repeated in their occupied slots.
    Tables are admitted only as complete rendered rows; prose retains the
    existing character budget. Unsupported tables are skipped, never sliced.
    """
    pieces: list[str] = []
    remaining = budget
    omitted = False
    compacted = False
    position = 0
    for match in _TABLE.finditer(content):
        prose = content[position:match.start()]
        pieces.append(prose[:remaining])
        omitted |= len(prose) > remaining
        remaining -= min(len(prose), remaining)
        parser = _TableParser()
        parser.feed(match.group())
        parser.close()
        table = parser.render()
        if table is None:
            omitted = True
        else:
            compacted = True
            header = "\n[表格；合并单元格按行列展开]\n"
            rows = table.splitlines()
            admitted = []
            used = len(header)
            notice = "[后续表格行已省略]\n"
            full_length = used + sum(len(row) + 1 for row in rows)
            row_budget = remaining if full_length <= remaining else remaining - len(notice)
            for row in rows:
                if used + len(row) + 1 > row_budget:
                    break
                admitted.append(row)
                used += len(row) + 1
            if admitted:
                pieces.append(header + "\n".join(admitted) + "\n")
                remaining -= used
                if len(admitted) != len(rows):
                    pieces.append(notice)
                    remaining -= len(notice)
            omitted |= len(admitted) != len(rows)
        position = match.end()
    tail = content[position:]
    pieces.append(tail[:remaining])
    omitted |= len(tail) > remaining
    return "".join(pieces).strip(), omitted, compacted
