/**
 * TF-IDF 向量化器 | TF-IDF Vectorizer
 *
 * 纯 JavaScript 实现，无外部依赖。
 * 支持中英文混合文本的词频统计和余弦相似度计算。
 *
 * Pure JavaScript implementation, no external dependencies.
 * Supports TF-IDF vectorization and cosine similarity for mixed Chinese-English text.
 *
 * @module @career-hub/rag-core/vectorizer
 * @author AchengBusiness
 * @version 1.0.0
 */

/**
 * 分词（支持中英文混合）| Tokenize text (supports Chinese-English mix)
 *
 * - 英文：按空白/标点分割后转小写 | English: split by whitespace/punctuation, lowercase
 * - 中文：按单字切分（unigram）| Chinese: single character unigram
 * - 过滤长度 < 1 的词元 | Filter tokens with length < 1
 *
 * @param {string} text - 输入文本 | Input text
 * @returns {string[]} 词元列表 | Token list
 */
export function tokenize(text) {
  if (!text || typeof text !== 'string') return [];

  const tokens = [];
  const englishRegex = /[a-zA-Z0-9]+/g;
  const chineseRegex = /[一-鿿㐀-䶿豈-﫿]/g;

  let match;

  while ((match = englishRegex.exec(text)) !== null) {
    const word = match[0].toLowerCase();
    if (word.length >= 1) tokens.push(word);
  }

  while ((match = chineseRegex.exec(text)) !== null) {
    tokens.push(match[0]);
  }

  return tokens;
}

/**
 * 计算词频（TF）| Compute term frequency (TF)
 *
 * 使用对数归一化：tf = 1 + log(count) for count > 0
 * Uses log normalization: tf = 1 + log(count) for count > 0
 *
 * @param {string[]} tokens - 词元列表 | Token list
 * @returns {Map<string, number>} 词元 → TF 值 | Token → TF value
 */
export function computeTF(tokens) {
  const counts = new Map();
  for (const token of tokens) {
    counts.set(token, (counts.get(token) || 0) + 1);
  }

  const tf = new Map();
  for (const [token, count] of counts) {
    tf.set(token, 1 + Math.log(count));
  }
  return tf;
}

/**
 * TF-IDF 向量化器类 | TF-IDF Vectorizer Class
 *
 * 工作流程 | Workflow:
 * 1. fit(documents) — 统计语料库，构建 IDF 表 | Analyze corpus, build IDF table
 * 2. transform(text) — 对单个文本计算 TF-IDF 向量（稀疏 Map）| Compute TF-IDF vector (sparse Map)
 * 3. cosineSimilarity(vecA, vecB) — 计算两向量余弦相似度 | Cosine similarity between two vectors
 *
 * @example
 * const v = new TFIDFVectorizer();
 * v.fit(['document one', '文档二']);
 * const vec = v.transform('document');
 * const sim = v.cosineSimilarity(vec, v.transform('one'));
 */
export class TFIDFVectorizer {
  constructor() {
    /** @type {Map<string, number>} 文档频率表 | Document frequency table */
    this._df = new Map();
    /** @type {number} 语料库文档数 | Number of documents in corpus */
    this._docCount = 0;
    /** @type {boolean} 是否已训练 | Whether fitted */
    this._fitted = false;
  }

  /**
   * 训练向量化器（建立 IDF 表）| Fit vectorizer (build IDF table)
   *
   * @param {string[]} documents - 文档文本列表 | List of document texts
   * @returns {TFIDFVectorizer} 支持链式调用 | Chainable
   */
  fit(documents) {
    if (!Array.isArray(documents) || documents.length === 0) {
      throw new Error('documents array cannot be empty');
    }

    this._df = new Map();
    this._docCount = documents.length;

    for (const doc of documents) {
      const tokens = tokenize(String(doc || ''));
      const uniqueTokens = new Set(tokens);
      for (const token of uniqueTokens) {
        this._df.set(token, (this._df.get(token) || 0) + 1);
      }
    }

    this._fitted = true;
    return this;
  }

  /**
   * 将文本转换为 TF-IDF 稀疏向量 | Transform text to TF-IDF sparse vector
   *
   * IDF = log((N + 1) / (df + 1)) + 1  (平滑处理 | smoothed)
   *
   * @param {string} text - 输入文本 | Input text
   * @returns {Map<string, number>} TF-IDF 稀疏向量 | TF-IDF sparse vector
   */
  transform(text) {
    if (!this._fitted) {
      throw new Error('Vectorizer not fitted, call fit() first');
    }

    const tokens = tokenize(String(text || ''));
    if (tokens.length === 0) return new Map();

    const tf = computeTF(tokens);
    const tfidf = new Map();

    for (const [token, tfVal] of tf) {
      const df = this._df.get(token) || 0;
      const idf = Math.log((this._docCount + 1) / (df + 1)) + 1;
      tfidf.set(token, tfVal * idf);
    }

    return tfidf;
  }

  /**
   * 计算两个稀疏向量的余弦相似度 | Compute cosine similarity between two sparse vectors
   *
   * @param {Map<string, number>} vecA - 向量 A | Vector A
   * @param {Map<string, number>} vecB - 向量 B | Vector B
   * @returns {number} 相似度 [0, 1] | Similarity [0, 1]
   */
  cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.size === 0 || vecB.size === 0) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (const [token, valA] of vecA) {
      normA += valA * valA;
      const valB = vecB.get(token);
      if (valB !== undefined) {
        dotProduct += valA * valB;
      }
    }

    for (const [, valB] of vecB) {
      normB += valB * valB;
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /** @returns {number} 词汇表大小 | Vocabulary size */
  get vocabularySize() { return this._df.size; }

  /** @returns {number} 训练文档数 | Number of training documents */
  get docCount() { return this._docCount; }

  /** @returns {boolean} 是否已训练 | Whether fitted */
  get fitted() { return this._fitted; }

  /**
   * 导出状态（序列化）| Export state (for serialization)
   * @returns {{df: Object, docCount: number}}
   */
  exportState() {
    return { df: Object.fromEntries(this._df), docCount: this._docCount };
  }

  /**
   * 从导出数据恢复状态（反序列化）| Restore state from exported data
   * @param {{df: Object, docCount: number}} state
   * @returns {TFIDFVectorizer}
   */
  importState(state) {
    if (!state || typeof state !== 'object') throw new Error('Invalid state data');
    this._df = new Map(Object.entries(state.df || {}));
    this._docCount = state.docCount || 0;
    this._fitted = this._docCount > 0;
    return this;
  }
}

export default TFIDFVectorizer;
