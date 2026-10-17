/* Versioned local history backups. Validate the entire file before any IDB write. */
(() => {
  "use strict";
  const MAX_BYTES = 64 * 1024 * 1024;
  const fail = () => { throw new Error("备份格式不正确或含无效记录，未导入任何内容。"); };
  const object = value => value && typeof value === "object" && !Array.isArray(value);
  const string = (value, max, empty = true) => {
    if (typeof value !== "string" || value.length > max || (!empty && !value.trim())) fail();
    return value;
  };
  const boolean = value => { if (typeof value !== "boolean") fail(); return value; };
  const seconds = value => {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) fail();
    return value;
  };
  const ids = value => {
    if (!Array.isArray(value) || value.length > 100 || value.some(id => !Number.isSafeInteger(id) || id <= 0)) fail();
    return value;
  };
  function source(value) {
    if (!object(value) || !Number.isSafeInteger(value.citation_id) || value.citation_id <= 0) fail();
    const result = {
      citation_id: value.citation_id, book: string(value.book, 200, false),
      section: string(value.section || "", 2000), excerpt: string(value.excerpt, 1024 * 1024),
      truncated: boolean(value.truncated ?? false), table_compacted: boolean(value.table_compacted ?? false),
    };
    for (const [key, max] of [["book_id", 503], ["chunk_id", 512]]) {
      if (value[key] !== undefined) result[key] = string(value[key], max);
    }
    if (value.source_sha256 != null) {
      if (typeof value.source_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.source_sha256)) fail();
      result.source_sha256 = value.source_sha256;
    }
    return result;
  }
  function snapshot(value) {
    if (!object(value) || !["answered", "retrieval_only", "no_evidence", "failed"].includes(value.status)) fail();
    const result = { status: value.status, answer: value.answer == null ? null : string(value.answer, 200000),
      message: value.message == null ? null : string(value.message, 4000), sources: [], conflicts: [], timing: {}, compute: {}, citation_integrity: null };
    if (value.status === "answered" && !result.answer?.trim()) fail();
    if (!Array.isArray(value.sources) || value.sources.length > 100) fail();
    result.sources = value.sources.map(source);
    if (new Set(result.sources.map(item => item.citation_id)).size !== result.sources.length) fail();
    if (value.conflicts !== undefined) {
      if (!Array.isArray(value.conflicts) || value.conflicts.length > 50) fail();
      result.conflicts = value.conflicts.map(item => string(item, 500));
    }
    if (value.timing !== undefined) {
      if (!object(value.timing)) fail();
      for (const key of ["retrieval_seconds", "generation_seconds", "first_token_seconds", "total_seconds"]) {
        if (value.timing[key] !== undefined) result.timing[key] = seconds(value.timing[key]);
      }
    }
    if (value.compute !== undefined) {
      if (!object(value.compute)) fail();
      for (const key of ["embedding", "reranker"]) {
        const stage = value.compute[key];
        if (stage == null) continue;
        if (!object(stage) || !["local", "remote", "mixed", "unknown"].includes(stage.backend)
            || !["cpu", "cuda", "mps", "mixed", "unknown"].includes(stage.device)
            || !["Windows", "Darwin", "Linux", "mixed", "unknown"].includes(stage.platform)) fail();
        result.compute[key] = { backend: stage.backend, device: stage.device, platform: stage.platform,
          elapsed_seconds: stage.elapsed_seconds == null ? null : seconds(stage.elapsed_seconds),
          fallback_used: boolean(stage.fallback_used ?? false) };
      }
    }
    if (value.citation_integrity != null) {
      const citation = value.citation_integrity;
      if (!object(citation) || !["linked", "missing", "invalid"].includes(citation.status)) fail();
      result.citation_integrity = { status: citation.status, cited: ids(citation.cited || []), unknown: ids(citation.unknown || []) };
    }
    return result;
  }
  function record(value) {
    if (!object(value) || typeof value.id !== "string" || !/^[\w-]{1,128}$/.test(value.id)
        || typeof value.created_at !== "string" || !Number.isFinite(Date.parse(value.created_at))) fail();
    const result = { id: value.id, created_at: new Date(value.created_at).toISOString(),
      query: string(value.query, 500, false), book_id: string(value.book_id, 503, false),
      book_label: string(value.book_label, 200, false), favorite: boolean(value.favorite),
      note: string(value.note ?? "", 4000), result: snapshot(value.result) };
    if (value.note_updated_at != null) {
      if (typeof value.note_updated_at !== "string" || !Number.isFinite(Date.parse(value.note_updated_at))) fail();
      result.note_updated_at = new Date(value.note_updated_at).toISOString();
    }
    if (value.review != null) {
      const review = value.review;
      if (!object(review) || !Number.isSafeInteger(review.count) || review.count < 1
          || !["again", "familiar"].includes(review.last_rating)) fail();
      for (const key of ["last_reviewed_at", "next_review_at"]) {
        if (typeof review[key] !== "string" || !Number.isFinite(Date.parse(review[key]))) fail();
      }
      result.review = {count: review.count, last_rating: review.last_rating,
        last_reviewed_at: new Date(review.last_reviewed_at).toISOString(),
        next_review_at: new Date(review.next_review_at).toISOString()};
    }
    return result;
  }
  function normalize(value) {
    if (!object(value) || value.format !== "rag-qa-history" || ![1, 2].includes(value.version)
        || !Array.isArray(value.records)) fail();
    if (value.records.length > 5000) throw new Error("单个备份最多 5000 条记录，请按教材筛选后分批备份。");
    const records = value.records.map(record);
    if (new Set(records.map(item => item.id)).size !== records.length) fail();
    return records;
  }
  function parse(text) {
    if (new Blob([text]).size > MAX_BYTES) throw new Error("单个备份最多 64 MB，请分教材备份。");
    try { return normalize(JSON.parse(text)); }
    catch (error) { if (error instanceof SyntaxError) fail(); throw error; }
  }
  function serialize(records) {
    const canonical = normalize({ format: "rag-qa-history", version: 2, records });
    const text = JSON.stringify({ format: "rag-qa-history", version: 2,
      exported_at: new Date().toISOString(), records: canonical }, null, 2);
    if (new Blob([text]).size > MAX_BYTES) throw new Error("备份超过 64 MB，请按教材筛选后分批备份。");
    return text;
  }
  window.RagHistoryBackup = Object.freeze({ parse, serialize, maxBytes: MAX_BYTES });
})();
