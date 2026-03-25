/**
 * 文本分块器 | Text Chunker
 *
 * 将长文档分割为适合检索的小块，支持三种分块策略：
 * - 固定大小重叠分块（fixed）
 * - 按段落分块（paragraph）
 * - 按 Markdown 标题分块（heading）
 *
 * Splits long documents into retrieval-friendly chunks using three strategies:
 * - Fixed-size overlapping chunks (fixed)
 * - Paragraph-based chunking (paragraph)
 * - Markdown heading-based chunking (heading)
 *
 * @module @career-hub/rag-core/chunker
 * @author AchengBusiness
 * @version 1.0.0
 */

/**
 * 按固定大小（含重叠）分块 | Chunk text by fixed size with overlap
 *
 * @param {string} text - 待分块文本 | Text to chunk
 * @param {number} [chunkSize=500] - 每块字符数 | Characters per chunk
 * @param {number} [overlap=50] - 重叠字符数 | Overlap characters
 * @returns {Array<{text: string, startIndex: number, endIndex: number, chunkIndex: number}>}
 */
export function chunkText(text, chunkSize = 500, overlap = 50) {
  if (!text || typeof text !== 'string') return [];
  const trimmed = text.trim();
  if (trimmed.length === 0) return [];

  if (chunkSize <= 0) throw new Error('chunkSize must be > 0');
  if (overlap < 0) throw new Error('overlap cannot be negative');
  if (overlap >= chunkSize) throw new Error('overlap must be < chunkSize');

  const chunks = [];
  let start = 0;
  let chunkIndex = 0;

  while (start < trimmed.length) {
    const end = Math.min(start + chunkSize, trimmed.length);
    const chunkContent = trimmed.slice(start, end).trim();

    if (chunkContent.length > 0) {
      chunks.push({ text: chunkContent, startIndex: start, endIndex: end, chunkIndex });
      chunkIndex++;
    }

    if (end >= trimmed.length) break;
    start = end - overlap;
  }

  return chunks;
}

/**
 * 按段落分块 | Chunk text by paragraphs
 *
 * 以空行（两个连续换行）作为段落分隔符，超过 maxChunkSize 的段落会进一步切分。
 * Splits by blank lines; paragraphs over maxChunkSize are further split.
 *
 * @param {string} text - 待分块文本 | Text to chunk
 * @param {number} [maxChunkSize=1000] - 单块最大字符数 | Max characters per chunk
 * @returns {Array<{text: string, paragraphIndex: number, chunkIndex: number}>}
 */
export function chunkByParagraph(text, maxChunkSize = 1000) {
  if (!text || typeof text !== 'string') return [];
  const trimmed = text.trim();
  if (trimmed.length === 0) return [];

  const paragraphs = trimmed.split(/
{2,}/);
  const chunks = [];
  let chunkIndex = 0;

  paragraphs.forEach((para, paraIdx) => {
    const cleaned = para.trim();
    if (!cleaned) return;

    if (cleaned.length <= maxChunkSize) {
      chunks.push({ text: cleaned, paragraphIndex: paraIdx, chunkIndex });
      chunkIndex++;
    } else {
      let start = 0;
      while (start < cleaned.length) {
        const end = Math.min(start + maxChunkSize, cleaned.length);
        const subChunk = cleaned.slice(start, end).trim();
        if (subChunk.length > 0) {
          chunks.push({ text: subChunk, paragraphIndex: paraIdx, chunkIndex });
          chunkIndex++;
        }
        start = end;
      }
    }
  });

  return chunks;
}

/**
 * 按 Markdown 标题分块 | Chunk text by Markdown headings
 *
 * 识别 #、##、### 等标题行，以每个标题为起点创建一个块。
 * Identifies heading lines and creates a chunk starting at each heading.
 *
 * @param {string} text - Markdown 文本 | Markdown text
 * @returns {Array<{text: string, heading: string, level: number, chunkIndex: number}>}
 */
export function chunkByHeading(text) {
  if (!text || typeof text !== 'string') return [];
  const trimmed = text.trim();
  if (trimmed.length === 0) return [];

  const lines = trimmed.split('
');
  const headingRegex = /^(#{1,6})\s+(.+)/;
  const sections = [];
  let currentHeading = '';
  let currentLevel = 0;
  let currentLines = [];

  for (const line of lines) {
    const match = line.match(headingRegex);
    if (match) {
      if (currentLines.length > 0 || currentHeading) {
        const bodyText = currentLines.join('
').trim();
        if (bodyText.length > 0 || currentHeading) {
          sections.push({ heading: currentHeading, level: currentLevel, body: bodyText });
        }
      }
      currentHeading = match[2].trim();
      currentLevel = match[1].length;
      currentLines = [];
    } else {
      currentLines.push(line);
    }
  }

  if (currentHeading || currentLines.length > 0) {
    const bodyText = currentLines.join('
').trim();
    if (bodyText.length > 0 || currentHeading) {
      sections.push({ heading: currentHeading, level: currentLevel, body: bodyText });
    }
  }

  return sections
    .filter((s) => s.heading || s.body)
    .map((s, i) => ({
      text: s.heading ? `${s.heading}
${s.body}`.trim() : s.body,
      heading: s.heading,
      level: s.level,
      chunkIndex: i,
    }));
}

export default { chunkText, chunkByParagraph, chunkByHeading };
