/**
 * 检索引擎 | Retrieval Engine
 *
 * 管理文档块的索引，提供基于 TF-IDF 余弦相似度的检索能力。
 * 支持文档的增删、懒更新索引和重排序。
 *
 * Manages the index of document chunks with TF-IDF cosine similarity retrieval.
 * Supports add/remove documents, lazy index updates, and re-ranking.
 *
 * @module @career-hub/rag-core/retriever
 * @author AchengBusiness
 * @version 1.0.0
 */

import { TFIDFVectorizer } from './vectorizer.js';

/**
 * Retriever — 检索引擎类 | Retrieval Engine Class
 *
 * 内部维护 chunks 列表和 TF-IDF 向量索引（懒更新：增删后标记为脏，查询时重建）。
 * Maintains chunk list and TF-IDF vector index (lazy update: dirty on changes, rebuilt on search).
 */
export class Retriever {
  /**
   * @param {Object} [options={}] - 配置项 | Options
   * @param {number} [options.topK=5] - 默认返回结果数 | Default number of results
   */
  constructor(options = {}) {
    /** @type {number} 默认 top-K | Default top-K */
    this.topK = options.topK ?? 5;

    /**
     * 块列表 | Chunk list
     * @type {Array<{id: string, docId: string, text: string, metadata: Object, chunkIndex: number}>}
     */
    this._chunks = [];

    /** @type {TFIDFVectorizer} */
    this._vectorizer = new TFIDFVectorizer();

    /**
     * 每个 chunk 的 TF-IDF 向量 | TF-IDF vector per chunk
     * @type {Map<string, number>[]}
     */
    this._vectors = [];

    /** @type {boolean} 是否需要重建索引 | Whether index needs rebuild */
    this._dirty = false;
  }

  /**
   * 将文档块加入索引 | Add a chunk to the index
   *
   * @param {Object} chunk - 块信息 | Chunk info
   * @param {string} chunk.id - 唯一块 ID | Unique chunk ID
   * @param {string} chunk.docId - 所属文档 ID | Parent document ID
   * @param {string} chunk.text - 块文本 | Chunk text
   * @param {Object} [chunk.metadata={}] - 附加元数据 | Additional metadata
   * @param {number} [chunk.chunkIndex=0] - 块在文档内的序号 | Chunk index within document
   */
  addChunk(chunk) {
    if (!chunk || !chunk.id || !chunk.docId || typeof chunk.text !== 'string') {
      throw new Error('chunk must have id, docId, and text');
    }
    this._removeChunkById(chunk.id);
    this._chunks.push({
      id: chunk.id,
      docId: chunk.docId,
      text: chunk.text,
      metadata: chunk.metadata || {},
      chunkIndex: chunk.chunkIndex ?? 0,
    });
    this._dirty = true;
  }

  /**
   * 批量添加块 | Batch add chunks
   * @param {Array} chunks
   */
  addChunks(chunks) {
    if (!Array.isArray(chunks)) throw new Error('chunks must be an array');
    for (const chunk of chunks) this.addChunk(chunk);
  }

  /**
   * 移除指定文档的所有块 | Remove all chunks for a document
   *
   * @param {string} docId - 文档 ID | Document ID
   * @returns {number} 移除的块数 | Number of removed chunks
   */
  removeDocument(docId) {
    const before = this._chunks.length;
    this._chunks = this._chunks.filter((c) => c.docId !== docId);
    const removed = before - this._chunks.length;
    if (removed > 0) this._dirty = true;
    return removed;
  }

  /**
   * 检索最相关的 K 个块 | Search for the top-K most relevant chunks
   *
   * @param {string} query - 查询文本 | Query text
   * @param {Object} [options={}] - 检索选项 | Search options
   * @param {number} [options.topK] - 返回结果数 | Number of results
   * @param {string} [options.docId] - 限定文档 ID（可选）| Restrict to document (optional)
   * @returns {Array<{chunk: Object, score: number, rank: number}>}
   */
  search(query, options = {}) {
    if (!query || typeof query !== 'string' || query.trim() === '') return [];
    if (this._chunks.length === 0) return [];

    if (this._dirty) this._rebuildIndex();

    const k = options.topK ?? this.topK;
    const queryVec = this._vectorizer.transform(query);

    let candidates = this._chunks.map((chunk, i) => ({
      chunk,
      score: this._vectorizer.cosineSimilarity(queryVec, this._vectors[i]),
    }));

    if (options.docId) {
      candidates = candidates.filter((c) => c.chunk.docId === options.docId);
    }

    candidates.sort((a, b) => b.score - a.score);
    return candidates.slice(0, k).map((c, rank) => ({ ...c, rank }));
  }

  /**
   * 重排序：按关键词命中数微调分数 | Re-rank: adjust scores by keyword hit count
   *
   * @param {Array} results - search() 返回的结果 | Results from search()
   * @param {string} query - 原查询文本 | Original query
   * @returns {Array} 重排序后的结果 | Re-ranked results
   */
  rerank(results, query) {
    if (!Array.isArray(results) || results.length === 0) return results;

    const queryTokens = new Set(
      query.toLowerCase().split(/[\s\p{P}]+/u).filter(Boolean)
    );

    const reranked = results.map((r) => {
      const chunkTokens = r.chunk.text.toLowerCase().split(/[\s\p{P}]+/u).filter(Boolean);
      const hits = chunkTokens.filter((t) => queryTokens.has(t)).length;
      return { ...r, score: r.score + hits * 0.01, keywordHits: hits };
    });

    reranked.sort((a, b) => b.score - a.score);
    return reranked.map((r, rank) => ({ ...r, rank }));
  }

  /** @returns {number} 索引中的块数 | Total chunk count */
  get size() { return this._chunks.length; }

  /** @returns {boolean} 是否已建立索引 | Whether index is built */
  get isIndexed() { return this._vectorizer.fitted && !this._dirty; }

  /**
   * 清空索引 | Clear the index
   */
  clear() {
    this._chunks = [];
    this._vectors = [];
    this._vectorizer = new TFIDFVectorizer();
    this._dirty = false;
  }

  /**
   * 导出索引状态（序列化）| Export index state (for serialization)
   * @returns {Object}
   */
  exportState() {
    if (this._dirty && this._chunks.length > 0) this._rebuildIndex();
    return {
      chunks: this._chunks,
      vectorizerState: this._vectorizer.fitted ? this._vectorizer.exportState() : null,
      topK: this.topK,
    };
  }

  /**
   * 从导出数据恢复索引（反序列化）| Restore index from exported data
   * @param {Object} state
   * @returns {Retriever}
   */
  importState(state) {
    if (!state || typeof state !== 'object') throw new Error('Invalid state data');
    this._chunks = Array.isArray(state.chunks) ? state.chunks : [];
    this.topK = state.topK ?? this.topK;

    if (state.vectorizerState) {
      this._vectorizer.importState(state.vectorizerState);
      this._vectors = this._chunks.map((chunk) => this._vectorizer.transform(chunk.text));
      this._dirty = false;
    } else {
      this._vectors = [];
      this._dirty = this._chunks.length > 0;
    }

    return this;
  }

  // ── Private ────────────────────────────────────────────────────

  /** @private */
  _rebuildIndex() {
    if (this._chunks.length === 0) {
      this._vectors = [];
      this._dirty = false;
      return;
    }
    const texts = this._chunks.map((c) => c.text);
    this._vectorizer = new TFIDFVectorizer();
    this._vectorizer.fit(texts);
    this._vectors = texts.map((t) => this._vectorizer.transform(t));
    this._dirty = false;
  }

  /** @private */
  _removeChunkById(id) {
    const idx = this._chunks.findIndex((c) => c.id === id);
    if (idx !== -1) {
      this._chunks.splice(idx, 1);
      this._dirty = true;
    }
  }
}

export default Retriever;
