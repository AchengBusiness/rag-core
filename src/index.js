/**
 * @career-hub/rag-core — 共享RAG知识库模块
 * Shared RAG Knowledge Base Module
 *
 * 轻量级、自包含、内存 TF-IDF 检索增强生成知识库。
 * 无外部依赖，支持中英文，可跨 Career-Hub 生态系统复用。
 *
 * Lightweight, self-contained, in-memory TF-IDF Retrieval-Augmented Generation knowledge base.
 * No external dependencies, supports Chinese and English,
 * designed for reuse across the Career-Hub ecosystem.
 *
 * 生态系统引用 | Ecosystem references:
 *   media-ops   → https://github.com/AchengBusiness/media-ops
 *   ops-flow    → https://github.com/AchengBusiness/ops-flow
 *   sales-crm   → https://github.com/AchengBusiness/sales-crm
 *   pm-toolkit  → https://github.com/AchengBusiness/pm-toolkit
 *
 * @module @career-hub/rag-core
 * @author AchengBusiness
 * @version 1.0.0
 *
 * @example
 * import RAGKnowledgeBase from '@career-hub/rag-core';
 *
 * const kb = new RAGKnowledgeBase({ chunkSize: 300, topK: 3 });
 *
 * // 添加文档 | Add documents
 * kb.addDocument('doc1', '产品路线图是规划产品发展方向的工具', { source: 'pm-guide' });
 * kb.addDocuments([
 *   { id: 'doc2', content: 'Follow-up best practices...', metadata: { category: 'sales' } },
 * ]);
 *
 * // 查询 | Query
 * const results = kb.query('路线图规划', { topK: 3 });
 * console.log(results[0].text, results[0].score);
 *
 * // 统计 | Stats
 * const stats = kb.getStats();
 * console.log(stats.documentCount, stats.chunkCount);
 *
 * // 持久化 | Persist
 * const snapshot = kb.exportKnowledge();
 * const kb2 = new RAGKnowledgeBase();
 * kb2.importKnowledge(snapshot);
 */

import { chunkText, chunkByParagraph, chunkByHeading } from './chunker.js';
import { TFIDFVectorizer, tokenize, computeTF } from './vectorizer.js';
import { Retriever } from './retriever.js';

/**
 * 分块策略常量 | Chunk strategy constants
 * @enum {string}
 */
export const ChunkStrategy = {
  /** 固定大小重叠分块 | Fixed-size overlapping chunks */
  FIXED: 'fixed',
  /** 按段落分块 | Paragraph-based chunking */
  PARAGRAPH: 'paragraph',
  /** 按 Markdown 标题分块 | Markdown heading-based chunking */
  HEADING: 'heading',
};

/**
 * RAGKnowledgeBase — 知识库主类 | Main Knowledge Base Class
 *
 * 封装文档管理、文本分块、TF-IDF 向量检索，提供统一高层 API。
 * Encapsulates document management, text chunking, and TF-IDF vector retrieval
 * with a unified high-level API.
 */
export class RAGKnowledgeBase {
  /**
   * 创建 RAGKnowledgeBase 实例 | Create a RAGKnowledgeBase instance
   *
   * @param {Object} [config={}] - 配置选项 | Configuration options
   * @param {string} [config.chunkStrategy='paragraph'] - 分块策略 | Chunking strategy
   * @param {number} [config.chunkSize=500] - 每块最大字符数 | Max characters per chunk
   * @param {number} [config.chunkOverlap=50] - 块间重叠字符数（仅 fixed 策略）| Chunk overlap (fixed only)
   * @param {number} [config.topK=5] - 默认返回结果数 | Default top-K results
   */
  constructor(config = {}) {
    this._config = {
      chunkStrategy: config.chunkStrategy ?? ChunkStrategy.PARAGRAPH,
      chunkSize: config.chunkSize ?? 500,
      chunkOverlap: config.chunkOverlap ?? 50,
      topK: config.topK ?? 5,
    };

    /**
     * 文档元数据存储：docId → { id, content, metadata, chunkIds, addedAt }
     * @type {Map<string, Object>}
     */
    this._documents = new Map();

    /**
     * 块索引：chunkId → { docId, chunkIndex, text }
     * @type {Map<string, Object>}
     */
    this._chunkMap = new Map();

    /** @type {Retriever} */
    this._retriever = new Retriever({ topK: this._config.topK });

    /** @type {boolean} 是否需要重建检索器索引 | Whether retriever needs rebuild */
    this._dirty = false;
  }

  // ── 文档管理 | Document Management ───────────────────────────────────────────

  /**
   * 添加单个文档到知识库 | Add a single document to the knowledge base
   *
   * 自动按配置的策略分块，并加入 TF-IDF 检索索引。
   * Automatically chunks by configured strategy and indexes for TF-IDF retrieval.
   *
   * @param {string} id - 文档唯一 ID（重复时覆盖）| Unique document ID (overwrite on duplicate)
   * @param {string} content - 文档文本内容 | Document text content
   * @param {Object} [metadata={}] - 附加元数据（来源、标签等）| Additional metadata
   * @returns {number} 产生的块数量 | Number of chunks produced
   * @throws {Error} id 或 content 无效时 | When id or content is invalid
   *
   * @example
   * kb.addDocument('sales-tips', '销售技巧内容...', { title: '销售技巧', category: 'sales' });
   */
  addDocument(id, content, metadata = {}) {
    if (!id || typeof id !== 'string' || id.trim() === '') {
      throw new Error('Document ID must be a non-empty string');
    }
    if (!content || typeof content !== 'string' || content.trim() === '') {
      throw new Error('Document content must be a non-empty string');
    }

    if (this._documents.has(id)) {
      this.removeDocument(id);
    }

    const chunks = this._chunkContent(content);
    const chunkIds = [];

    for (let i = 0; i < chunks.length; i++) {
      const chunkId = `${id}__chunk_${i}`;
      const chunkText = typeof chunks[i] === 'string' ? chunks[i] : chunks[i].text;
      this._chunkMap.set(chunkId, { docId: id, chunkIndex: i, text: chunkText });
      this._retriever.addChunk({
        id: chunkId,
        docId: id,
        text: chunkText,
        metadata: { ...metadata, docId: id },
        chunkIndex: i,
      });
      chunkIds.push(chunkId);
    }

    this._documents.set(id, {
      id,
      content,
      metadata,
      chunkIds,
      addedAt: new Date().toISOString(),
    });

    return chunks.length;
  }

  /**
   * 批量添加文档 | Batch add documents
   *
   * @param {Array<{id: string, content: string, metadata?: Object}>} docs - 文档数组 | Document array
   * @returns {number} 成功添加的文档数 | Number of successfully added documents
   */
  addDocuments(docs) {
    if (!Array.isArray(docs)) throw new Error('docs must be an array');
    let count = 0;
    for (const doc of docs) {
      this.addDocument(doc.id, doc.content, doc.metadata || {});
      count++;
    }
    return count;
  }

  /**
   * 从知识库中删除文档 | Remove a document from the knowledge base
   *
   * @param {string} id - 要删除的文档 ID | Document ID to remove
   * @returns {boolean} 是否成功删除 | Whether successfully removed
   */
  removeDocument(id) {
    const doc = this._documents.get(id);
    if (!doc) return false;

    for (const chunkId of doc.chunkIds) {
      this._chunkMap.delete(chunkId);
    }
    this._retriever.removeDocument(id);
    this._documents.delete(id);
    return true;
  }

  // ── 检索 | Retrieval ──────────────────────────────────────────────────────────

  /**
   * 查询知识库，返回最相关的文档块 | Query the knowledge base for most relevant chunks
   *
   * @param {string} query - 查询文本 | Query text
   * @param {Object} [options={}] - 查询选项 | Query options
   * @param {number} [options.topK] - 返回结果数（覆盖默认值）| Top-K results (overrides default)
   * @param {boolean} [options.rerank=false] - 是否启用重排序 | Whether to enable re-ranking
   * @param {number} [options.minScore=0] - 最低相似度阈值 | Minimum similarity threshold
   * @returns {Array<{documentId: string, chunkId: string, chunkIndex: number, text: string, metadata: Object, score: number}>}
   *
   * @example
   * const results = kb.query('如何跟进客户', { topK: 3, minScore: 0.05 });
   * results.forEach(r => console.log(r.text, r.score));
   */
  query(query, options = {}) {
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      throw new Error('query must be a non-empty string');
    }
    if (this._documents.size === 0) return [];

    const topK = options.topK ?? this._config.topK;
    const rerank = options.rerank ?? false;
    const minScore = options.minScore ?? 0;

    let results = this._retriever.search(query.trim(), { topK });

    if (rerank) {
      results = this._retriever.rerank(results, query.trim());
    }

    return results
      .filter((r) => r.score >= minScore)
      .map((r) => ({
        documentId: r.chunk.docId,
        chunkId: r.chunk.id,
        chunkIndex: r.chunk.chunkIndex,
        text: r.chunk.text,
        metadata: r.chunk.metadata,
        score: r.score,
      }));
  }

  // ── 信息查询 | Stats & Info ───────────────────────────────────────────────────

  /**
   * 获取知识库统计信息 | Get knowledge base statistics
   *
   * @returns {{ documentCount: number, chunkCount: number, vocabularySize: number, isIndexed: boolean, config: Object }}
   */
  getStats() {
    return {
      documentCount: this._documents.size,
      chunkCount: this._chunkMap.size,
      vocabularySize: this._retriever.isIndexed
        ? this._retriever._vectorizer.vocabularySize
        : 0,
      isIndexed: this._retriever.isIndexed,
      config: { ...this._config },
    };
  }

  /**
   * 获取指定文档信息 | Get a specific document
   *
   * @param {string} id - 文档 ID | Document ID
   * @returns {Object|null}
   */
  getDocument(id) {
    return this._documents.get(id) || null;
  }

  /**
   * 列出所有文档 ID | List all document IDs
   * @returns {string[]}
   */
  listDocuments() {
    return Array.from(this._documents.keys());
  }

  // ── 序列化 | Serialization ────────────────────────────────────────────────────

  /**
   * 清空知识库所有数据 | Clear all data from the knowledge base
   */
  clear() {
    this._documents.clear();
    this._chunkMap.clear();
    this._retriever.clear();
    this._dirty = false;
  }

  /**
   * 导出知识库为可序列化对象（用于持久化）
   * Export the knowledge base as a serializable object (for persistence)
   *
   * @returns {Object} 可 JSON 序列化的快照 | JSON-serializable snapshot
   */
  exportKnowledge() {
    return {
      version: '1.0.0',
      config: { ...this._config },
      documents: Array.from(this._documents.values()).map((doc) => ({
        id: doc.id,
        content: doc.content,
        metadata: doc.metadata,
        addedAt: doc.addedAt,
      })),
      exportedAt: new Date().toISOString(),
    };
  }

  /**
   * 从导出数据恢复知识库 | Import a previously exported knowledge base
   *
   * @param {Object} data - exportKnowledge() 返回的数据 | Data from exportKnowledge()
   * @returns {RAGKnowledgeBase} this（链式调用）| this (chainable)
   */
  importKnowledge(data) {
    if (!data || typeof data !== 'object') throw new Error('import data must be a non-null object');

    this.clear();

    if (data.config) {
      this._config = { ...this._config, ...data.config };
    }

    for (const doc of (data.documents || [])) {
      if (doc && doc.id && doc.content) {
        this.addDocument(doc.id, doc.content, doc.metadata || {});
      }
    }

    return this;
  }

  // ── 别名（向后兼容）| Aliases (backward compatibility) ──────────────────────

  /** @deprecated 请使用 exportKnowledge() | Use exportKnowledge() instead */
  export() { return this.exportKnowledge(); }

  /** @deprecated 请使用 importKnowledge() | Use importKnowledge() instead */
  import(data) { return this.importKnowledge(data); }

  // ── 私有辅助 | Private helpers ────────────────────────────────────────────────

  /**
   * 根据配置的策略切分文本 | Chunk content by configured strategy
   * @param {string} content
   * @returns {Array}
   * @private
   */
  _chunkContent(content) {
    const { chunkStrategy, chunkSize, chunkOverlap } = this._config;

    switch (chunkStrategy) {
      case ChunkStrategy.FIXED:
        return chunkText(content, chunkSize, chunkOverlap);

      case ChunkStrategy.HEADING:
        return chunkByHeading(content);

      case ChunkStrategy.PARAGRAPH:
      default:
        return chunkByParagraph(content, chunkSize);
    }
  }
}

// ── 子模块导出 | Sub-module exports ──────────────────────────────────────────
export { chunkText, chunkByParagraph, chunkByHeading } from './chunker.js';
export { TFIDFVectorizer, tokenize, computeTF } from './vectorizer.js';
export { Retriever } from './retriever.js';

export default RAGKnowledgeBase;
