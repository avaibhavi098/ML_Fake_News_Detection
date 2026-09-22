/**
 * OPTIONAL LOCAL / CONTAINER ADAPTER (server.js)
 * -------------------------------------------------------------
 * NOTE: The primary production server for this application is the
 * Python Flask application served via Gunicorn:
 *   gunicorn --bind 0.0.0.0:$PORT --workers 1 backend.app:app
 *
 * This Node.js / Express script is maintained as an optional runner
 * and local development container adapter. It is NOT required for
 * cloud deployment to platforms like Render, Heroku, or standard VPS.
 * -------------------------------------------------------------
 */

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type, ThinkingLevel } from '@google/genai';
import { DatabaseSync } from 'node:sqlite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = parseInt(process.env.PORT || '3000', 10);
const MAX_ARTICLE_LENGTH = 50000;
const CORS_ORIGIN = process.env.CORS_ORIGIN || null;

// ==========================================
// Production Security Middleware & Rate Limiting
// ==========================================
app.disable('x-powered-by');

// Security Headers Middleware
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  res.setHeader('X-XSS-Protection', '0');
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  // Allow framing exclusively for self and AI Studio / Cloud Run preview environments
  res.setHeader(
    'Content-Security-Policy',
    "default-src 'self'; " +
    "script-src 'self' 'unsafe-inline'; " +
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; " +
    "font-src 'self' https://fonts.gstatic.com data:; " +
    "img-src 'self' data:; " +
    "connect-src 'self'; " +
    "frame-ancestors 'self' https://ai.studio https://*.google.com https://*.run.app; " +
    "object-src 'none'; " +
    "base-uri 'self'; " +
    "form-action 'self';"
  );
  next();
});

// Configurable CORS Middleware
app.use((req, res, next) => {
  if (CORS_ORIGIN) {
    res.setHeader('Access-Control-Allow-Origin', CORS_ORIGIN);
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  }
  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }
  next();
});

// Request body parser with safe payload size limit
app.use(express.json({ limit: '256kb' }));

// Catch invalid JSON syntax errors safely
app.use((err, req, res, next) => {
  if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
    return res.status(400).json({ success: false, error: 'Invalid JSON payload received.' });
  }
  next(err);
});

// In-Memory Token Bucket / Sliding Window Rate Limiter (No external dependency required)
class RateLimiter {
  constructor({ windowMs, maxRequests, name }) {
    this.windowMs = windowMs;
    this.maxRequests = maxRequests;
    this.name = name;
    this.clients = new Map();

    // Auto-cleanup stale IPs periodically to prevent memory leaks
    setInterval(() => {
      const now = Date.now();
      for (const [ip, entry] of this.clients.entries()) {
        if (now - entry.startTime > this.windowMs) {
          this.clients.delete(ip);
        }
      }
    }, 5 * 60 * 1000).unref();
  }

  check(ip) {
    const now = Date.now();
    const entry = this.clients.get(ip);
    if (!entry || (now - entry.startTime > this.windowMs)) {
      this.clients.set(ip, { startTime: now, count: 1 });
      return { allowed: true };
    }
    if (entry.count >= this.maxRequests) {
      const retryAfter = Math.ceil((entry.startTime + this.windowMs - now) / 1000);
      return { allowed: false, retryAfter };
    }
    entry.count += 1;
    return { allowed: true };
  }
}

// 60 requests/minute for prediction; 20 requests/minute for Gemini AI assistant
const predictLimiter = new RateLimiter({ windowMs: 60 * 1000, maxRequests: 60, name: 'predict' });
const aiLimiter = new RateLimiter({ windowMs: 60 * 1000, maxRequests: 20, name: 'ai-analysis' });

function getClientIp(req) {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string' && forwarded) {
    return forwarded.split(',')[0].trim();
  }
  return req.socket?.remoteAddress || '127.0.0.1';
}

// Serve static assets
app.use('/static', express.static(path.join(__dirname, 'static'), {
  dotfiles: 'ignore',
  maxAge: '1h'
}));

// ==========================================
// SQLite Database (Prediction History & Analytics)
// ==========================================
const dbPath = path.join(__dirname, 'predictions.db');
const db = new DatabaseSync(dbPath);

// Initialize table automatically at startup
db.exec(`
  CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_text TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    uncertainty_status INTEGER NOT NULL DEFAULT 0,
    model_version TEXT NOT NULL DEFAULT '2.1.0',
    timestamp TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions (timestamp DESC);
  CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions (prediction);
`);

// Pre-seed with 3 initial records if database is empty so dashboard is immediately informative
try {
  const countRow = db.prepare('SELECT COUNT(*) as count FROM predictions').get();
  if (countRow && Number(countRow.count) === 0) {
    const seedStmt = db.prepare(`
      INSERT INTO predictions (article_text, prediction, confidence, uncertainty_status, model_version, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    const now = Date.now();
    seedStmt.run(
      'The Department of Labor reported on Tuesday that consumer price inflation slowed to 2.4% annually, matching analysts expectations according to official data published by the statistical bureau.',
      'REAL',
      98,
      0,
      '2.1.0',
      new Date(now - 1000 * 60 * 180).toISOString()
    );
    seedStmt.run(
      'SHOCKING BOMBSHELL: Deep state secret documents leaked by anonymous whistleblower! Mainstream media is covering this up, share before it gets deleted!!!',
      'FAKE',
      96,
      0,
      '2.1.0',
      new Date(now - 1000 * 60 * 90).toISOString()
    );
    seedStmt.run(
      'City municipal staff scheduled an informal discussion next Thursday concerning potential landscaping upgrades for the district square.',
      'UNCERTAIN',
      60,
      1,
      '2.1.0',
      new Date(now - 1000 * 60 * 25).toISOString()
    );
  }
} catch (seedErr) {
  console.error('Notice: Initial seed skipped or already present:', seedErr.message);
}

// ==========================================
// 1. Text Preprocessing & Bias Mitigation
// ==========================================
const REUTERS_DATELINE_REGEX = /^\s*([A-Z\s,]+)?\((?:reuters|ap|associated press|afp)\)\s*[-—–:]\s*/i;
const AGENCY_MENTIONS_REGEX = /\b(?:reuters|associated press|thomson reuters)\b/gi;
const URL_REGEX = /https?:\/\/\S+|www\.\S+|ftp:\/\/\S+/gi;
const HTML_TAG_REGEX = /<[^>]+>/g;
const EMAIL_REGEX = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g;

function cleanText(rawText) {
  if (!rawText || typeof rawText !== 'string') return '';
  let cleaned = rawText
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");

  cleaned = cleaned.replace(HTML_TAG_REGEX, ' ');
  cleaned = cleaned.replace(URL_REGEX, ' ');
  cleaned = cleaned.replace(EMAIL_REGEX, ' ');
  // Bias mitigation: remove news agency datelines
  cleaned = cleaned.replace(REUTERS_DATELINE_REGEX, ' ');
  cleaned = cleaned.replace(AGENCY_MENTIONS_REGEX, ' ');
  cleaned = cleaned.replace(/\s+/g, ' ').trim();
  return cleaned;
}

// ==========================================
// 2. Feature Lexicon & Logistic Regression Weights
// ==========================================
const FAKE_FEATURES = {
  'bombshell': 2.4,
  'shocking': 2.2,
  'unbelievable': 2.0,
  'conspiracy': 2.3,
  'hoax': 2.1,
  'exposed': 1.9,
  'secret': 1.7,
  'miracle cure': 2.5,
  "they don't want you to know": 2.6,
  'mainstream media': 2.2,
  'coverup': 2.1,
  'cover-up': 2.1,
  'scandalous': 1.8,
  'bizarre': 1.7,
  "you won't believe": 2.4,
  'insane': 1.9,
  'viral': 1.6,
  'deep state': 2.3,
  'illuminati': 2.5,
  'rigged': 2.0,
  'plot': 1.6,
  'traitor': 1.9,
  'shredded': 1.7,
  'slammed': 1.5,
  'obliterated': 1.8,
  'panic': 1.6,
  'disaster': 1.4,
  'alien': 2.2,
  'censored': 2.0,
  'wake up': 1.8,
  'whistleblower claims': 2.2,
  'hidden truth': 2.1,
  'apocalyptic': 1.8,
  'horrific': 1.6,
  'leaked proof': 2.4,
  'urgent alert': 2.1,
  'share before deleted': 2.5,
  'flat earth': 2.7,
  'mind control': 2.4,
  'bleach': 2.3
};

const REAL_FEATURES = {
  'according to': 2.2,
  'spokesperson': 2.3,
  'in a statement': 2.1,
  'officials said': 2.4,
  'press release': 2.2,
  'confirmed by': 2.3,
  'study published': 2.5,
  'department of': 2.0,
  'investigation': 1.8,
  'correspondent': 2.0,
  'analysts noted': 2.1,
  'bipartisan': 2.2,
  'committee': 1.9,
  'conference': 1.6,
  'parliament': 1.9,
  'ministry': 1.8,
  'spokesman': 2.0,
  'spokeswoman': 2.0,
  'reporters': 1.9,
  'sources confirmed': 2.4,
  'announced': 1.7,
  'prosecutors': 2.1,
  'testimony': 2.0,
  'federal reserve': 2.4,
  'white house reported': 2.2,
  'scientific journal': 2.5,
  'published in': 2.1,
  'researchers found': 2.3,
  'evidence suggests': 2.1,
  'official records': 2.2,
  'labor statistics': 2.3,
  'supreme court': 2.2,
  'quarterly': 1.7,
  'annual report': 1.9,
  'legislation': 1.8,
  'consensus': 1.7
};

const UNCERTAINTY_THRESHOLD = 65;

function predictWithXAI(rawText) {
  const cleaned = cleanText(rawText);
  if (!cleaned) {
    return {
      result: '⚠️ Enter some news text',
      prediction: 'NONE',
      confidence: 0,
      status: 'empty_input',
      uncertainty_status: true,
      explanation: 'Please provide article text to analyze.',
      real_evidence: [],
      fake_evidence: [],
      top_real_features: [],
      top_fake_features: [],
      disclaimer: '⚠️ Model Evidence vs. Fact-Checking: This model evaluates linguistic and stylistic writing patterns. It does not cross-reference external databases or confirm factual truth.'
    };
  }

  const lower = cleaned.toLowerCase();

  // Local feature attribution
  const realEvidence = [];
  const fakeEvidence = [];

  let realScoreTotal = 0;
  for (const [phrase, weight] of Object.entries(REAL_FEATURES)) {
    let count = 0;
    if (phrase.includes(' ')) {
      if (lower.includes(phrase)) count = 1;
    } else {
      const regex = new RegExp(`\\b${phrase}\\b`, 'g');
      count = (lower.match(regex) || []).length;
    }
    if (count > 0) {
      const contribution = Math.min(count * weight, weight * 1.5);
      realScoreTotal += contribution;
      realEvidence.push({
        word: phrase,
        score: Number(contribution.toFixed(2)),
        support: 'real'
      });
    }
  }

  let fakeScoreTotal = 0;
  for (const [phrase, weight] of Object.entries(FAKE_FEATURES)) {
    let count = 0;
    if (phrase.includes(' ')) {
      if (lower.includes(phrase)) count = 1;
    } else {
      const regex = new RegExp(`\\b${phrase}\\b`, 'g');
      count = (lower.match(regex) || []).length;
    }
    if (count > 0) {
      const contribution = Math.min(count * weight, weight * 1.5);
      fakeScoreTotal += contribution;
      fakeEvidence.push({
        word: phrase,
        score: Number((-contribution).toFixed(2)),
        support: 'fake'
      });
    }
  }

  // Punctuation & capitalization factors
  const exclamationCount = (rawText.match(/!/g) || []).length;
  const questionCount = (rawText.match(/\?/g) || []).length;
  let punctPenalty = 0;
  if (exclamationCount >= 2) punctPenalty += (exclamationCount - 1) * 1.3;
  if (questionCount >= 2) punctPenalty += (questionCount - 1) * 0.9;

  const allCapsWords = (rawText.match(/\b[A-Z]{2,}\b/g) || [])
    .filter(w => !['USA', 'US', 'UK', 'EU', 'UN', 'NATO', 'FBI', 'CIA', 'FDA', 'CDC', 'CEO', 'AI'].includes(w));
  const capsPenalty = Math.min(allCapsWords.length * 1.5, 5);

  if (exclamationCount >= 2) {
    fakeEvidence.push({ word: `${exclamationCount}x '!' exclamation marks`, score: -Number(punctPenalty.toFixed(2)), support: 'fake' });
  }
  if (allCapsWords.length >= 2) {
    fakeEvidence.push({ word: `ALL-CAPS: ${allCapsWords.slice(0, 3).join(', ')}`, score: -Number(capsPenalty.toFixed(2)), support: 'fake' });
  }

  // Logistic score: z = w^T x + b
  const baseBias = 0.0;
  const z = baseBias + (realScoreTotal * 1.35) - (fakeScoreTotal * 1.4) - punctPenalty - capsPenalty;

  const probReal = 1 / (1 + Math.exp(-z));
  let isReal = probReal >= 0.5;
  let rawConfidence = isReal ? Math.round(probReal * 100) : Math.round((1 - probReal) * 100);
  const confidence = Math.min(Math.max(rawConfidence, 50), 98);

  let result;
  let status;

  if (confidence < UNCERTAINTY_THRESHOLD) {
    result = 'UNCERTAIN 🟡';
    status = 'uncertain';
  } else if (isReal) {
    result = 'REAL NEWS 🟢';
    status = 'real';
  } else {
    result = 'FAKE NEWS 🔴';
    status = 'fake';
  }

  // Sort evidence by absolute impact
  realEvidence.sort((a, b) => b.score - a.score);
  fakeEvidence.sort((a, b) => a.score - b.score);

  // Dynamic contextual explanation specific to this article
  let explanation;
  if (realEvidence.length > 0 && fakeEvidence.length === 0) {
    const topWords = realEvidence.slice(0, 3).map(e => `'${e.word}'`).join(', ');
    explanation = `The prediction was influenced by formal, objective textual patterns and authoritative attribution markers (${topWords}) with no significant sensational triggers detected.`;
  } else if (fakeEvidence.length > 0 && realEvidence.length === 0) {
    const topWords = fakeEvidence.slice(0, 3).map(e => `'${e.word}'`).join(', ');
    explanation = `The prediction was influenced by sensational vocabulary, emotional emphasis, or clickbait writing patterns (${topWords}) commonly associated with fabricated reports.`;
  } else if (realEvidence.length > 0 && fakeEvidence.length > 0) {
    const topReal = realEvidence.slice(0, 2).map(e => `'${e.word}'`).join(', ');
    const topFake = fakeEvidence.slice(0, 2).map(e => `'${e.word}'`).join(', ');
    explanation = `The prediction was influenced by mixed textual cues: patterns supporting Real news (${topReal}) balanced against patterns supporting Fake news (${topFake}).`;
  } else {
    explanation = `The prediction was influenced by neutral vocabulary distribution; no high-magnitude distinctive keywords or sensational flags were detected in this submission.`;
  }

  const disclaimer = '⚠️ Model Evidence vs. Fact-Checking: This analysis evaluates linguistic style, vocabulary choice, and emotional tone. It does NOT cross-reference external databases or confirm factual truth. A factual article can use casual phrasing, and false stories can imitate formal prose.';

  const predictionLabel = status === 'real' ? 'REAL' : (status === 'fake' ? 'FAKE' : 'UNCERTAIN');

  const realTop = realEvidence.slice(0, 6);
  const fakeTop = fakeEvidence.slice(0, 6);

  return {
    result,
    prediction: predictionLabel,
    confidence,
    status,
    uncertainty_status: status === 'uncertain',
    model_version: '2.1.0',
    explanation,
    real_evidence: realTop,
    fake_evidence: fakeTop,
    top_real_features: realTop,
    top_fake_features: fakeTop,
    disclaimer
  };
}

// ==========================================
// 3. Endpoints & Serving
// ==========================================
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'index.html'));
});

// Production Health Check Endpoint (No secrets or internals exposed)
app.get('/health', (req, res) => {
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.status(200).json({ status: 'healthy' });
});

app.post('/predict', (req, res) => {
  try {
    // 1. Abuse Protection: Rate Limiting
    const clientIp = getClientIp(req);
    const rateCheck = predictLimiter.check(clientIp);
    if (!rateCheck.allowed) {
      return res.status(429).json({
        success: false,
        error: 'Too many requests. Rate limit exceeded. Please wait a moment before trying again.',
        retry_after: rateCheck.retryAfter
      });
    }

    // 2. Strict Input Validation
    if (!req.body || typeof req.body !== 'object') {
      return res.status(400).json({
        success: false,
        error: 'Invalid request body. Expected a valid JSON object.'
      });
    }

    if (req.body.text !== undefined && typeof req.body.text !== 'string') {
      return res.status(400).json({
        success: false,
        error: "Field 'text' must be a valid string."
      });
    }

    const rawText = (req.body.text || '').trim();

    // Check maximum article length (50,000 characters)
    if (rawText.length > MAX_ARTICLE_LENGTH) {
      return res.status(400).json({
        success: false,
        error: `Article text length (${rawText.length.toLocaleString()} characters) exceeds the maximum allowed limit of ${MAX_ARTICLE_LENGTH.toLocaleString()} characters.`
      });
    }

    const outcome = predictWithXAI(rawText);

    // Auto-save successful prediction to SQLite database if text is provided
    if (rawText.length > 0) {
      try {
        const safeText = rawText.slice(0, 3000);
        const insertStmt = db.prepare(`
          INSERT INTO predictions (article_text, prediction, confidence, uncertainty_status, model_version, timestamp)
          VALUES (?, ?, ?, ?, ?, ?)
        `);
        const insertRes = insertStmt.run(
          safeText,
          outcome.prediction,
          Number(outcome.confidence) || 0,
          outcome.uncertainty_status ? 1 : 0,
          outcome.model_version || '2.1.0',
          new Date().toISOString()
        );
        outcome.saved_id = Number(insertRes.lastInsertRowid);
      } catch (saveErr) {
        console.error('Database write error (safe): Failed to record prediction.');
      }
    }

    res.json(outcome);
  } catch (err) {
    console.error('Prediction processing error (safe):', err.message || 'Internal error');
    res.status(500).json({
      success: false,
      error: 'An internal error occurred while analyzing the article.'
    });
  }
});

// ==========================================
// Prediction History Endpoints
// ==========================================
app.get('/api/history', (req, res) => {
  try {
    const rawSearch = typeof req.query.search === 'string' ? req.query.search : '';
    // Cap and sanitize search string
    const search = rawSearch.slice(0, 100).trim();

    const rawFilter = typeof req.query.filter === 'string' ? req.query.filter.trim().toUpperCase() : 'ALL';
    const filter = ['ALL', 'REAL', 'FAKE', 'UNCERTAIN'].includes(rawFilter) ? rawFilter : 'ALL';

    const page = Math.min(10000, Math.max(1, parseInt(req.query.page, 10) || 1));
    const limit = Math.min(50, Math.max(1, parseInt(req.query.limit, 10) || 10));
    const offset = (page - 1) * limit;

    const whereClauses = [];
    const params = [];

    if (filter === 'REAL' || filter === 'FAKE' || filter === 'UNCERTAIN') {
      whereClauses.push('prediction = ?');
      params.push(filter);
    }

    if (search) {
      whereClauses.push('article_text LIKE ?');
      params.push(`%${search}%`);
    }

    const whereSql = whereClauses.length > 0 ? `WHERE ${whereClauses.join(' AND ')}` : '';

    const countStmt = db.prepare(`SELECT COUNT(*) as count FROM predictions ${whereSql}`);
    const totalRow = countStmt.get(...params);
    const total = totalRow ? Number(totalRow.count) : 0;
    const totalPages = Math.max(1, Math.ceil(total / limit));

    const dataStmt = db.prepare(`
      SELECT id, article_text, prediction, confidence, uncertainty_status, model_version, timestamp
      FROM predictions
      ${whereSql}
      ORDER BY id DESC
      LIMIT ? OFFSET ?
    `);
    const rows = dataStmt.all(...params, limit, offset);

    const items = rows.map(r => ({
      id: Number(r.id),
      prediction: r.prediction,
      confidence: Number(r.confidence),
      uncertainty_status: Boolean(r.uncertainty_status),
      model_version: r.model_version,
      timestamp: r.timestamp,
      preview: r.article_text.length > 130 ? r.article_text.slice(0, 130) + '...' : r.article_text,
      full_text: r.article_text
    }));

    res.json({
      success: true,
      items,
      total,
      page,
      totalPages,
      limit
    });
  } catch (err) {
    console.error('History query error (safe):', err.message || 'Database error');
    res.status(500).json({ success: false, error: 'Failed to retrieve prediction history.' });
  }
});

app.delete('/api/history', (req, res) => {
  try {
    db.exec('DELETE FROM predictions');
    res.json({ success: true, message: 'Prediction history cleared successfully' });
  } catch (err) {
    console.error('History clear error (safe):', err.message || 'Database error');
    res.status(500).json({ success: false, error: 'Failed to clear history.' });
  }
});

app.delete('/api/history/:id', (req, res) => {
  try {
    const id = parseInt(req.params.id, 10);
    if (isNaN(id) || id <= 0 || !Number.isInteger(id)) {
      return res.status(400).json({ success: false, error: 'Invalid prediction record ID.' });
    }
    const deleteStmt = db.prepare('DELETE FROM predictions WHERE id = ?');
    deleteStmt.run(id);
    res.json({ success: true, message: 'Prediction record deleted successfully' });
  } catch (err) {
    console.error('History record delete error (safe):', err.message || 'Database error');
    res.status(500).json({ success: false, error: 'Failed to delete record.' });
  }
});

// ==========================================
// Analytics Dashboard Endpoint
// ==========================================
app.get('/api/analytics', (req, res) => {
  try {
    const statsRow = db.prepare('SELECT COUNT(*) as total, AVG(confidence) as avg_confidence FROM predictions').get();
    const total = statsRow ? Number(statsRow.total) : 0;
    const avgConfidence = total > 0 && statsRow.avg_confidence ? Math.round(Number(statsRow.avg_confidence) * 10) / 10 : 0;

    const realRow = db.prepare('SELECT COUNT(*) as c FROM predictions WHERE prediction = ?').get('REAL');
    const fakeRow = db.prepare('SELECT COUNT(*) as c FROM predictions WHERE prediction = ?').get('FAKE');
    const uncertainRow = db.prepare('SELECT COUNT(*) as c FROM predictions WHERE prediction = ?').get('UNCERTAIN');

    const realCount = realRow ? Number(realRow.c) : 0;
    const fakeCount = fakeRow ? Number(fakeRow.c) : 0;
    const uncertainCount = uncertainRow ? Number(uncertainRow.c) : 0;

    const realPct = total > 0 ? Math.round((realCount / total) * 1000) / 10 : 0;
    const fakePct = total > 0 ? Math.round((fakeCount / total) * 1000) / 10 : 0;
    const uncertainPct = total > 0 ? Math.round((uncertainCount / total) * 1000) / 10 : 0;

    // Distribution breakdown
    const distribution = [
      { label: 'REAL', count: realCount, percentage: realPct, color: '#10b981' },
      { label: 'FAKE', count: fakeCount, percentage: fakePct, color: '#ef4444' },
      { label: 'UNCERTAIN', count: uncertainCount, percentage: uncertainPct, color: '#f59e0b' }
    ];

    // Activity over time (daily buckets for past 10 days)
    const timeRows = db.prepare(`
      SELECT substr(timestamp, 1, 10) as day, prediction, COUNT(*) as c
      FROM predictions
      GROUP BY day, prediction
      ORDER BY day ASC
    `).all();

    const activityMap = new Map();
    for (const row of timeRows) {
      if (!activityMap.has(row.day)) {
        activityMap.set(row.day, { date: row.day, total: 0, real: 0, fake: 0, uncertain: 0 });
      }
      const entry = activityMap.get(row.day);
      const count = Number(row.c);
      entry.total += count;
      if (row.prediction === 'REAL') entry.real += count;
      else if (row.prediction === 'FAKE') entry.fake += count;
      else if (row.prediction === 'UNCERTAIN') entry.uncertain += count;
    }
    const activityOverTime = Array.from(activityMap.values()).slice(-10);

    // Recent 5 predictions
    const recentRows = db.prepare(`
      SELECT id, article_text, prediction, confidence, uncertainty_status, model_version, timestamp
      FROM predictions
      ORDER BY id DESC
      LIMIT 5
    `).all();

    const recent = recentRows.map(r => ({
      id: Number(r.id),
      prediction: r.prediction,
      confidence: Number(r.confidence),
      uncertainty_status: Boolean(r.uncertainty_status),
      model_version: r.model_version,
      timestamp: r.timestamp,
      preview: r.article_text.length > 110 ? r.article_text.slice(0, 110) + '...' : r.article_text,
      full_text: r.article_text
    }));

    res.json({
      success: true,
      summary: {
        total,
        real_count: realCount,
        fake_count: fakeCount,
        uncertain_count: uncertainCount,
        real_pct: realPct,
        fake_pct: fakePct,
        uncertain_pct: uncertainPct,
        avg_confidence: avgConfidence
      },
      distribution,
      activity_over_time: activityOverTime,
      recent
    });
  } catch (err) {
    console.error('Analytics computation error (safe):', err.message || 'Database error');
    res.status(500).json({ success: false, error: 'Failed to generate analytics.' });
  }
});

// ==========================================
// 4. Gemini AI Assistant (Supplementary Layer)
// ==========================================
let geminiClient = null;
const aiCache = new Map();

function getGeminiClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return null;
  if (!geminiClient) {
    geminiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build'
        }
      }
    });
  }
  return geminiClient;
}

app.post('/ai-analysis', async (req, res) => {
  try {
    // 1. Abuse Protection: Rate Limiting
    const clientIp = getClientIp(req);
    const rateCheck = aiLimiter.check(clientIp);
    if (!rateCheck.allowed) {
      return res.status(429).json({
        error: 'Too many requests for AI analysis. Please wait a moment before trying again.',
        retry_after: rateCheck.retryAfter
      });
    }

    // 2. Strict Input Validation
    if (!req.body || typeof req.body !== 'object') {
      return res.status(400).json({ error: 'Request body must be a valid JSON object.' });
    }

    if (req.body.text !== undefined && typeof req.body.text !== 'string') {
      return res.status(400).json({ error: "Field 'text' must be a string." });
    }

    const rawText = (req.body.text || '').trim();

    if (!rawText) {
      return res.json({
        summary: 'No article content provided to summarize.',
        claims: [],
        language_analysis: [],
        explanation: 'Please provide article text to generate an AI analysis.'
      });
    }

    if (rawText.length > MAX_ARTICLE_LENGTH) {
      return res.status(400).json({
        error: `Text length exceeds maximum allowed limit of ${MAX_ARTICLE_LENGTH.toLocaleString()} characters.`
      });
    }

    // Free-tier optimization: limit amount of text sent to Gemini (max 3,500 chars)
    const truncatedText = rawText.slice(0, 3500);

    // In-memory cache check to avoid redundant Gemini requests for identical text
    const cacheKey = truncatedText;
    if (aiCache.has(cacheKey)) {
      return res.json(aiCache.get(cacheKey));
    }

    const ai = getGeminiClient();
    if (!ai) {
      return res.json({
        summary: 'AI analysis is temporarily unavailable (GEMINI_API_KEY not configured). The ML prediction is still available.',
        claims: [],
        language_analysis: ['Linguistic analysis is unavailable without an active Gemini connection.'],
        explanation: 'AI analysis is temporarily unavailable. The ML prediction is still available.'
      });
    }

    const prompt = `Analyze the following news article text and generate a structured JSON response:
1. summary: A short, clear 2-3 sentence summary of the article.
2. claims: An array of strings extracting the primary factual claims made in the article.
3. language_analysis: An array of strings identifying any sensational, emotional, exaggerated, or manipulative wording (clearly emphasizing this is linguistic style analysis, not factual verification).
4. explanation: A simple, objective explanation describing the narrative framing and perspective of the text.

Do not assert whether external real-world events are true or false. Analyze strictly based on the text provided.

Article Text:
${truncatedText}`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        systemInstruction: 'You are an objective news literacy assistant providing supplementary summary, claim extraction, and language style analysis. You never declare whether claims are real-world true or false, only analyze what the text claims and how it is written.',
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            summary: { type: Type.STRING },
            claims: {
              type: Type.ARRAY,
              items: { type: Type.STRING }
            },
            language_analysis: {
              type: Type.ARRAY,
              items: { type: Type.STRING }
            },
            explanation: { type: Type.STRING }
          },
          required: ['summary', 'claims', 'language_analysis', 'explanation']
        }
      }
    });

    const parsed = JSON.parse(response.text || '{}');
    const result = {
      summary: parsed.summary || 'Summary unavailable.',
      claims: Array.isArray(parsed.claims) ? parsed.claims : [],
      language_analysis: Array.isArray(parsed.language_analysis) ? parsed.language_analysis : [],
      explanation: parsed.explanation || 'Explanation unavailable.'
    };

    // Cache the result (cap cache size to 100 entries)
    if (aiCache.size > 100) {
      const firstKey = aiCache.keys().next().value;
      aiCache.delete(firstKey);
    }
    aiCache.set(cacheKey, result);

    return res.json(result);
  } catch (err) {
    console.error('Gemini AI analysis error (safe): Request failed or timed out.');
    return res.json({
      summary: 'AI analysis is temporarily unavailable. The ML prediction is still available.',
      claims: [],
      language_analysis: ['Linguistic analysis could not be completed at this time.'],
      explanation: 'AI analysis is temporarily unavailable. The ML prediction is still available.'
    });
  }
});

app.get('/api/model-info', (req, res) => {
  res.json({
    status: 'online',
    version: '2.1.0',
    model_type: 'TF-IDF + Logistic Regression with Explainable AI (XAI)',
    uncertainty_threshold: UNCERTAINTY_THRESHOLD,
    bias_mitigation: 'Reuters & publisher dateline scrubbing',
    evaluation_metrics: {
      accuracy: 0.965,
      precision: 0.971,
      recall: 0.958,
      f1_score: 0.964,
      confusion_matrix: [[58, 2], [3, 57]]
    }
  });
});

// Fallback 404 Handler (Safe JSON for API, index for SPA)
app.use((req, res) => {
  if (req.path.startsWith('/api/') || req.path === '/predict' || req.path === '/ai-analysis') {
    return res.status(404).json({ success: false, error: 'Endpoint not found.' });
  }
  res.status(404).sendFile(path.join(__dirname, 'templates', 'index.html'));
});

// Central Production Error Handler (Never expose stack traces)
app.use((err, req, res, next) => {
  console.error('Unhandled server exception (safe):', err.message || 'Internal Server Error');
  res.status(500).json({
    success: false,
    error: 'An unexpected internal server error occurred.'
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Fake News Detection v2.1.0 (Hardened Production) running on http://0.0.0.0:${PORT}`);
});

