/* Small escaped Markdown renderer used by the chat page. No network dependencies. */
(() => {
  "use strict";
  const escapeHtml = (text) => text.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[character]));

  // Runs on already-escaped text, so every tag it emits is one it chose to emit.
  function inline(text) {
    const plain = value => value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/【参考资料\s*(\d+)】/g, '<button type="button" class="cite" data-cite="$1">资料 $1</button>');
    let result = "", offset = 0;
    for (const match of text.matchAll(/`([^`]+)`/g)) {
      result += plain(text.slice(offset, match.index)) + "<code>" + match[1] + "</code>";
      offset = match.index + match[0].length;
    }
    return result + plain(text.slice(offset));
  }

  function tableCells(line) {
    if (!line.includes("|")) return [];
    const cells = [], value = line.trim();
    let text = "";
    for (let index = 0; index < value.length; index++) {
      if (value[index] === "\\" && value[index + 1] === "|") { text += "|"; index++; }
      else if (value[index] === "|") { cells.push(text.trim()); text = ""; }
      else text += value[index];
    }
    cells.push(text.trim());
    if (value.startsWith("|")) cells.shift();
    if (value.endsWith("|") && !value.endsWith("\\|")) cells.pop();
    return cells;
  }

  const codeBlock = lines => '<div class="code-block"><button class="code-copy" type="button" data-copy-code>复制代码</button><pre><code>' + lines.join("\n") + '</code></pre></div>';

  function renderMarkdown(source) {
    const out = [];
    let list = null, code = null, fence = "", paragraph = [];
    const lines = escapeHtml(source).split("\n");
    const flushParagraph = () => {
      if (paragraph.length) { out.push("<p>" + inline(paragraph.join("<br>")) + "</p>"); paragraph = []; }
    };
    const closeList = () => { if (list) { out.push("</" + list + ">"); list = null; } };
    for (let index = 0; index < lines.length; index++) {
      const line = lines[index];
      if (code !== null) {
        const closing = line.match(/^\s*(`{3,}|~{3,})\s*$/);
        if (closing && closing[1][0] === fence[0] && closing[1].length >= fence.length) { out.push(codeBlock(code)); code = null; }
        else code.push(line);
        continue;
      }
      const opening = line.match(/^\s*(`{3,}|~{3,})/);
      if (opening) { flushParagraph(); closeList(); code = []; fence = opening[1]; continue; }
      const headers = tableCells(line), separators = tableCells(lines[index + 1] || "");
      if (headers.length && separators.length === headers.length && separators.every(cell => /^:?-{3,}:?$/.test(cell))) {
        flushParagraph(); closeList();
        const align = separators.map(cell => cell.startsWith(":") && cell.endsWith(":") ? "center" : cell.endsWith(":") ? "right" : "left");
        out.push('<div class="table-scroll" tabindex="0" role="region" aria-label="回答中的表格"><table><thead><tr>');
        headers.forEach((cell, column) => out.push(`<th scope="col" style="text-align:${align[column]}">${inline(cell)}</th>`));
        out.push("</tr></thead><tbody>");
        index++;
        while (index + 1 < lines.length) {
          const cells = tableCells(lines[index + 1]);
          if (cells.length !== headers.length) break;
          index++;
          out.push("<tr>");
          cells.forEach((cell, column) => out.push(`<td style="text-align:${align[column]}">${inline(cell)}</td>`));
          out.push("</tr>");
        }
        out.push("</tbody></table></div>");
        continue;
      }
      const heading = line.match(/^\s*(#{1,6})\s+(.*)$/);
      if (heading) {
        flushParagraph(); closeList();
        const level = Math.min(heading[1].length + 1, 3);
        out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
        continue;
      }
      const bullet = line.match(/^\s*[-*]\s+(.*)$/);
      const numbered = bullet ? null : line.match(/^\s*(\d+)[.)]\s+(.*)$/);
      if (bullet || numbered) {
        flushParagraph();
        const kind = bullet ? "ul" : "ol";
        if (list !== kind) {
          closeList();
          // Keep the model's own number. A paragraph between items closes the list,
          // and a fresh <ol> would otherwise restart every item at 1.
          out.push(numbered ? `<ol start="${numbered[1]}">` : "<ul>");
          list = kind;
        }
        out.push("<li>" + inline(bullet ? bullet[1] : numbered[2]) + "</li>");
        continue;
      }
      if (!line.trim()) { flushParagraph(); closeList(); continue; }
      closeList();
      paragraph.push(line);
    }
    if (code !== null) out.push(codeBlock(code));
    flushParagraph(); closeList();
    return out.join("");
  }

  window.RagMarkdown = Object.freeze({render: renderMarkdown});
})();
