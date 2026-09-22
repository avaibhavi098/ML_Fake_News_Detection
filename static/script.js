/**
 * Fake News Detection System - Modern Frontend UI Controller
 * Preserves 100% of ML, XAI, Gemini, and Database logic.
 */

const SAMPLES = {
    real: "The Department of Labor reported on Tuesday that consumer price inflation slowed to 2.4% annually, matching analysts' expectations according to official data published by the statistical bureau.",
    fake: "SHOCKING BOMBSHELL: Deep state secret documents leaked by anonymous whistleblower! Mainstream media is covering this up, share before it gets deleted!!!",
    uncertain: "City municipal staff scheduled an informal discussion next Thursday concerning potential landscaping upgrades for the district square."
};

/**
 * Text Counters for Word & Character count
 */
function updateTextCounters() {
    const textarea = document.getElementById("newsText");
    const wordElem = document.getElementById("wordCount");
    const charElem = document.getElementById("charCount");
    if (!textarea) return;

    const text = textarea.value || "";
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const chars = text.length;

    if (wordElem) wordElem.innerText = `${words} ${words === 1 ? 'word' : 'words'}`;
    if (charElem) charElem.innerText = `${chars.toLocaleString()} ${chars === 1 ? 'character' : 'characters'}`;
}

/**
 * Populate Sample or Clear Textarea
 */
function fillSample(type) {
    const textarea = document.getElementById("newsText");
    if (!textarea) return;

    if (type === "clear") {
        textarea.value = "";
        updateTextCounters();
        const resultBox = document.getElementById("resultBox");
        if (resultBox) resultBox.classList.add("hidden");
        const saveNotice = document.getElementById("saveStatusNotice");
        if (saveNotice) saveNotice.classList.add("hidden");
        const loadingElem = document.getElementById("aiLoading");
        const contentBox = document.getElementById("aiContent");
        const errorBox = document.getElementById("aiErrorBox");
        if (loadingElem) loadingElem.classList.add("hidden");
        if (contentBox) contentBox.classList.add("hidden");
        if (errorBox) errorBox.classList.add("hidden");
        textarea.focus();
    } else if (SAMPLES[type]) {
        textarea.value = SAMPLES[type];
        updateTextCounters();
        checkNews();
    }
}

/**
 * Safely escape string to prevent XSS
 */
function escapeHtml(text) {
    if (!text && text !== 0) return "";
    const div = document.createElement("div");
    div.textContent = String(text);
    return div.innerHTML;
}

// In-memory text store to prevent string-interpolation injection in inline onclick handlers
const historyTextStore = new Map();

function inspectHistoryById(id) {
    const text = historyTextStore.get(Number(id));
    if (text) {
        loadIntoDetector(text);
    }
}

/**
 * In-Context Word Highlighting for XAI Evidence
 */
function renderHighlightedText(rawText, realEvidence, fakeEvidence) {
    let text = escapeHtml(rawText);

    // Filter and sort tokens by length descending so longer compound phrases match first
    const tokens = [];
    (realEvidence || []).forEach(item => {
        if (item.word && !item.word.includes("!") && !item.word.startsWith("ALL-CAPS")) {
            tokens.push({ word: item.word, type: "real" });
        }
    });
    (fakeEvidence || []).forEach(item => {
        if (item.word && !item.word.includes("!") && !item.word.startsWith("ALL-CAPS")) {
            tokens.push({ word: item.word, type: "fake" });
        }
    });

    tokens.sort((a, b) => b.word.length - a.word.length);

    tokens.forEach(token => {
        const escaped = token.word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const regex = new RegExp(`(^|[^a-zA-Z0-9])(${escaped})([^a-zA-Z0-9]|$)`, "gi");
        text = text.replace(regex, `$1<mark class="hl-${token.type}">$2</mark>$3`);
    });

    return text;
}

/**
 * Primary News Verification Check
 */
function checkNews() {
    const textarea = document.getElementById("newsText");
    const rawText = textarea ? textarea.value : "";
    const text = rawText.trim();
    const btn = document.getElementById("btnCheck");

    if (!text) {
        if (textarea) {
            textarea.focus();
            textarea.style.borderColor = "#ef4444";
            setTimeout(() => { textarea.style.borderColor = ""; }, 1500);
        }
        return;
    }

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="ai-spinner" style="width:16px;height:16px;border-width:2px;border-top-color:#ffffff;"></span> <span>Analyzing Evidence...</span>`;
    }

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            const errorMsg = data.error || (res.status === 429 ? "Too many requests. Please wait a moment." : "An error occurred.");
            throw new Error(errorMsg);
        }
        return data;
    })
    .then(data => {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<span class="btn-icon">⚡</span> <span class="btn-text">Analyze News Article</span>`;
        }

        const resultBox = document.getElementById("resultBox");
        if (resultBox) resultBox.classList.remove("hidden");

        const predictionElem = document.getElementById("prediction");
        const barElem = document.getElementById("confidenceBar");
        const textElem = document.getElementById("confidenceText");
        const badgeElem = document.getElementById("confidenceBadge");
        const verdictCard = document.getElementById("verdictCard");
        const versionElem = document.getElementById("modelVersionTag");
        const verdictDetailElem = document.getElementById("verdictDetailText");
        const explainElem = document.getElementById("explainText");
        const realListElem = document.getElementById("realEvidenceList");
        const fakeListElem = document.getElementById("fakeEvidenceList");
        const highlightedElem = document.getElementById("highlightedText");
        const disclaimerElem = document.getElementById("disclaimerText");

        if (predictionElem) predictionElem.innerText = data.result;
        if (textElem) textElem.innerText = data.confidence + "%";
        if (barElem) {
            barElem.style.width = data.confidence + "%";
            barElem.className = "confidence-fill";
        }
        if (versionElem && data.model_version) {
            versionElem.innerText = "Model v" + data.model_version;
        }

        // Clean previous verdict card classes
        if (verdictCard) {
            verdictCard.classList.remove("verdict-real", "verdict-fake", "verdict-uncertain");
        }

        if (data.result.includes("REAL")) {
            if (barElem) barElem.classList.add("fill-real");
            if (verdictCard) verdictCard.classList.add("verdict-real");
            if (badgeElem) {
                badgeElem.innerText = "High Credibility";
                badgeElem.className = "confidence-badge real";
            }
            if (verdictDetailElem) {
                verdictDetailElem.innerText = "Formal attribution and objective journalistic patterns detected.";
            }
        } else if (data.result.includes("FAKE")) {
            if (barElem) barElem.classList.add("fill-fake");
            if (verdictCard) verdictCard.classList.add("verdict-fake");
            if (badgeElem) {
                badgeElem.innerText = "High Deception Risk";
                badgeElem.className = "confidence-badge fake";
            }
            if (verdictDetailElem) {
                verdictDetailElem.innerText = "Sensational or manipulative vocabulary patterns detected in text.";
            }
        } else if (data.result.includes("UNCERTAIN")) {
            if (barElem) barElem.classList.add("fill-uncertain");
            if (verdictCard) verdictCard.classList.add("verdict-uncertain");
            if (badgeElem) {
                badgeElem.innerText = "Uncertain / Neutral";
                badgeElem.className = "confidence-badge uncertain";
            }
            if (verdictDetailElem) {
                verdictDetailElem.innerText = "Vocabulary is neutral or ambiguous; confidence score near decision boundary.";
            }
        }

        // Auto-save banner feedback & tab badge count update
        const saveNotice = document.getElementById("saveStatusNotice");
        if (saveNotice) saveNotice.classList.remove("hidden");
        updateHistoryBadge();

        // Dynamic human-readable XAI explanation
        if (explainElem && data.explanation) {
            explainElem.innerText = data.explanation;
        }

        // Render Real Evidence tags
        if (realListElem) {
            if (data.real_evidence && data.real_evidence.length > 0) {
                realListElem.innerHTML = data.real_evidence.map(e => `
                    <span class="feature-badge real" title="Positive weight: +${Math.abs(e.score)}">
                        ${escapeHtml(e.word)}
                        <span class="feature-score">+${Math.abs(e.score)}</span>
                    </span>
                `).join("");
            } else {
                realListElem.innerHTML = `<span class="no-evidence-text">No distinct journalistic markers detected</span>`;
            }
        }

        // Render Fake Evidence tags
        if (fakeListElem) {
            if (data.fake_evidence && data.fake_evidence.length > 0) {
                fakeListElem.innerHTML = data.fake_evidence.map(e => `
                    <span class="feature-badge fake" title="Negative weight: -${Math.abs(e.score)}">
                        ${escapeHtml(e.word)}
                        <span class="feature-score">-${Math.abs(e.score)}</span>
                    </span>
                `).join("");
            } else {
                fakeListElem.innerHTML = `<span class="no-evidence-text">No distinct sensational markers detected</span>`;
            }
        }

        // Render In-Context Highlighted text
        if (highlightedElem) {
            highlightedElem.innerHTML = renderHighlightedText(rawText, data.real_evidence, data.fake_evidence);
        }

        // Render disclaimer
        if (disclaimerElem && data.disclaimer) {
            disclaimerElem.innerText = data.disclaimer;
        }

        // Trigger supplementary AI Assistant analysis
        if (text) {
            fetchAiAnalysis(false);
        }
    })
    .catch((err) => {
        console.error("Prediction error:", err);
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<span class="btn-icon">⚡</span> <span class="btn-text">Analyze News Article</span>`;
        }
        const resultBox = document.getElementById("resultBox");
        if (resultBox) resultBox.classList.remove("hidden");
        const prediction = document.getElementById("prediction");
        if (prediction) prediction.innerText = err.message ? `⚠️ ${err.message}` : "⚠️ Unable to complete request";
    });
}

// Client-side cache to avoid redundant Gemini requests for identical text
const clientAiCache = new Map();

function fetchAiAnalysis(forceRefresh) {
    const textarea = document.getElementById("newsText");
    const rawText = textarea ? textarea.value : "";
    const text = (rawText || "").trim();

    const loadingElem = document.getElementById("aiLoading");
    const errorBox = document.getElementById("aiErrorBox");
    const errorMsg = document.getElementById("aiErrorMessage");
    const contentBox = document.getElementById("aiContent");
    const btnRun = document.getElementById("btnRunAi");

    if (!text) {
        if (loadingElem) loadingElem.classList.add("hidden");
        if (contentBox) contentBox.classList.add("hidden");
        if (errorBox) errorBox.classList.add("hidden");
        return;
    }

    if (!forceRefresh && clientAiCache.has(text)) {
        renderAiAnalysisData(clientAiCache.get(text));
        return;
    }

    if (loadingElem) loadingElem.classList.remove("hidden");
    if (errorBox) errorBox.classList.add("hidden");
    if (contentBox) contentBox.classList.add("hidden");
    if (btnRun) {
        btnRun.disabled = true;
        btnRun.innerText = "Evaluating...";
    }

    fetch("/ai-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(async res => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            const err = new Error(data.error || "AI assistant temporarily unavailable.");
            err.status = res.status;
            throw err;
        }
        return data;
    })
    .then(data => {
        if (loadingElem) loadingElem.classList.add("hidden");
        if (btnRun) {
            btnRun.disabled = false;
            btnRun.innerText = "🤖 Re-evaluate with AI";
        }

        const isUnavailable = !data || data.error || (data.summary && data.summary.includes("temporarily unavailable"));

        if (isUnavailable) {
            if (errorBox) errorBox.classList.remove("hidden");
            if (errorMsg) errorMsg.innerText = data.error || "AI analysis is temporarily unavailable. The ML prediction is fully available.";
            if (contentBox) contentBox.classList.add("hidden");
            return;
        }

        clientAiCache.set(text, data);
        renderAiAnalysisData(data);
    })
    .catch((err) => {
        if (loadingElem) loadingElem.classList.add("hidden");
        if (btnRun) {
            btnRun.disabled = false;
            btnRun.innerText = "🤖 Re-evaluate with AI";
        }
        if (errorBox) errorBox.classList.remove("hidden");
        if (errorMsg) {
            if (err.status === 429) {
                errorMsg.innerText = "Rate limit reached for AI Assistant. Please wait a moment.";
            } else {
                errorMsg.innerText = err.message || "AI analysis is temporarily unavailable. The ML prediction is fully available.";
            }
        }
        if (contentBox) contentBox.classList.add("hidden");
    });
}

function renderAiAnalysisData(data) {
    const errorBox = document.getElementById("aiErrorBox");
    const contentBox = document.getElementById("aiContent");
    const summaryElem = document.getElementById("aiSummary");
    const claimsElem = document.getElementById("aiClaims");
    const languageElem = document.getElementById("aiLanguage");
    const explainElem = document.getElementById("aiExplanation");

    if (errorBox) errorBox.classList.add("hidden");
    if (contentBox) contentBox.classList.remove("hidden");

    if (summaryElem) {
        summaryElem.innerText = data.summary || "Summary not available.";
    }

    if (claimsElem) {
        if (data.claims && Array.isArray(data.claims) && data.claims.length > 0) {
            claimsElem.innerHTML = data.claims.map(c => `<li>${escapeHtml(c)}</li>`).join("");
        } else {
            claimsElem.innerHTML = `<li>No explicit factual assertions extracted from the submitted text.</li>`;
        }
    }

    if (languageElem) {
        if (data.language_analysis && Array.isArray(data.language_analysis) && data.language_analysis.length > 0) {
            languageElem.innerHTML = data.language_analysis.map(l => `<li>${escapeHtml(l)}</li>`).join("");
        } else {
            languageElem.innerHTML = `<li>Neutral phrasing observed with standard lexical distribution.</li>`;
        }
    }

    if (explainElem) {
        explainElem.innerText = data.explanation || "No explanation available.";
    }
}

/* ==========================================
   Navigation & Tab Switching
   ========================================== */
let activeTab = "detector";

function switchTab(tabName) {
    activeTab = tabName;
    const tabs = ["detector", "history", "analytics"];

    tabs.forEach(name => {
        const capitalized = name.charAt(0).toUpperCase() + name.slice(1);
        const btn = document.getElementById(`tabBtn${capitalized}`);
        const view = document.getElementById(`view${capitalized}`);

        if (name === tabName) {
            if (btn) {
                btn.classList.add("active");
                btn.setAttribute("aria-selected", "true");
            }
            if (view) {
                view.classList.remove("hidden");
            }
        } else {
            if (btn) {
                btn.classList.remove("active");
                btn.setAttribute("aria-selected", "false");
            }
            if (view) {
                view.classList.add("hidden");
            }
        }
    });

    if (tabName === "analytics") {
        loadAnalytics();
    } else if (tabName === "history") {
        loadHistory(currentHistoryPage);
    }
}

/* ==========================================
   Analytics Dashboard
   ========================================== */
function loadAnalytics() {
    const btn = document.getElementById("btnRefreshAnalytics");
    if (btn) btn.innerHTML = `<span class="btn-icon">🔄</span> <span>Refreshing...</span>`;

    fetch("/api/analytics")
        .then(res => res.json())
        .then(data => {
            if (btn) btn.innerHTML = `<span class="btn-icon">🔄</span> <span>Refresh Metrics</span>`;
            renderAnalytics(data);
        })
        .catch(err => {
            console.error("Failed to load analytics:", err);
            if (btn) btn.innerHTML = `<span class="btn-icon">🔄</span> <span>Refresh Metrics</span>`;
        });
}

function renderAnalytics(data) {
    const summary = data.summary || {};
    const total = data.total_predictions !== undefined ? data.total_predictions : (summary.total || 0);
    const realCount = (data.category_counts && data.category_counts.REAL !== undefined) ? data.category_counts.REAL : (summary.real_count || 0);
    const fakeCount = (data.category_counts && data.category_counts.FAKE !== undefined) ? data.category_counts.FAKE : (summary.fake_count || 0);
    const uncertainCount = (data.category_counts && data.category_counts.UNCERTAIN !== undefined) ? data.category_counts.UNCERTAIN : (summary.uncertain_count || 0);

    const realPct = (data.category_percentages && data.category_percentages.REAL !== undefined) ? data.category_percentages.REAL : (summary.real_pct || 0);
    const fakePct = (data.category_percentages && data.category_percentages.FAKE !== undefined) ? data.category_percentages.FAKE : (summary.fake_pct || 0);
    const uncertainPct = (data.category_percentages && data.category_percentages.UNCERTAIN !== undefined) ? data.category_percentages.UNCERTAIN : (summary.uncertain_pct || 0);

    const avgConf = data.average_confidence !== undefined ? data.average_confidence : (summary.avg_confidence || 0);

    // 1. KPI Cards
    const totalElem = document.getElementById("kpiTotal");
    const realElem = document.getElementById("kpiReal");
    const realPctElem = document.getElementById("kpiRealPct");
    const fakeElem = document.getElementById("kpiFake");
    const fakePctElem = document.getElementById("kpiFakePct");
    const uncertainElem = document.getElementById("kpiUncertain");
    const uncertainPctElem = document.getElementById("kpiUncertainPct");
    const avgConfElem = document.getElementById("kpiAvgConfidence");

    if (totalElem) totalElem.innerText = total.toLocaleString();
    if (realElem) realElem.innerText = realCount.toLocaleString();
    if (realPctElem) realPctElem.innerText = realPct + "%";
    if (fakeElem) fakeElem.innerText = fakeCount.toLocaleString();
    if (fakePctElem) fakePctElem.innerText = fakePct + "%";
    if (uncertainElem) uncertainElem.innerText = uncertainCount.toLocaleString();
    if (uncertainPctElem) uncertainPctElem.innerText = uncertainPct + "%";
    if (avgConfElem) avgConfElem.innerText = avgConf + "%";

    // Update History Badge as well
    const badge = document.getElementById("historyCountBadge");
    if (badge) badge.innerText = total.toLocaleString();

    // 2. Stacked Distribution Bar
    const barReal = document.getElementById("barReal");
    const barFake = document.getElementById("barFake");
    const barUncertain = document.getElementById("barUncertain");

    if (total > 0) {
        if (barReal) barReal.style.width = realPct + "%";
        if (barFake) barFake.style.width = fakePct + "%";
        if (barUncertain) barUncertain.style.width = uncertainPct + "%";
    } else {
        if (barReal) barReal.style.width = "33.3%";
        if (barFake) barFake.style.width = "33.3%";
        if (barUncertain) barUncertain.style.width = "33.4%";
    }

    // 3. Legend Stats
    const legReal = document.getElementById("legendRealVal");
    const legFake = document.getElementById("legendFakeVal");
    const legUncertain = document.getElementById("legendUncertainVal");

    if (legReal) legReal.innerText = `${realCount} (${realPct}%)`;
    if (legFake) legFake.innerText = `${fakeCount} (${fakePct}%)`;
    if (legUncertain) legUncertain.innerText = `${uncertainCount} (${uncertainPct}%)`;

    // 4. SVG Donut Chart
    renderDonutChart(realCount, fakeCount, uncertainCount, total);

    // 5. Timeline Activity Chart
    renderActivityChart(data.activity_over_time || []);

    // 6. Recent Predictions Feed
    renderRecentActivity(data.recent || data.recent_predictions || []);
}

function renderDonutChart(real, fake, uncertain, total) {
    const svg = document.getElementById("donutSvg");
    const totalElem = document.getElementById("donutCenterTotal");
    if (totalElem) totalElem.innerText = total.toLocaleString();

    if (!svg) return;

    if (total === 0) {
        svg.innerHTML = `
            <circle cx="100" cy="100" r="70" fill="none" stroke="#1e293b" stroke-width="20" />
        `;
        return;
    }

    const radius = 70;
    const circumference = 2 * Math.PI * radius; // ~439.82

    const realArc = (real / total) * circumference;
    const fakeArc = (fake / total) * circumference;
    const uncertainArc = (uncertain / total) * circumference;

    let offset = 0;
    let svgHtml = "";

    if (real > 0) {
        svgHtml += `
            <circle cx="100" cy="100" r="${radius}" fill="none"
                stroke="#10b981" stroke-width="20"
                stroke-dasharray="${realArc} ${circumference}"
                stroke-dashoffset="${-offset}"
                style="transition: stroke-dasharray 0.4s ease;" />
        `;
        offset += realArc;
    }

    if (fake > 0) {
        svgHtml += `
            <circle cx="100" cy="100" r="${radius}" fill="none"
                stroke="#ef4444" stroke-width="20"
                stroke-dasharray="${fakeArc} ${circumference}"
                stroke-dashoffset="${-offset}"
                style="transition: stroke-dasharray 0.4s ease;" />
        `;
        offset += fakeArc;
    }

    if (uncertain > 0) {
        svgHtml += `
            <circle cx="100" cy="100" r="${radius}" fill="none"
                stroke="#f59e0b" stroke-width="20"
                stroke-dasharray="${uncertainArc} ${circumference}"
                stroke-dashoffset="${-offset}"
                style="transition: stroke-dasharray 0.4s ease;" />
        `;
    }

    svg.innerHTML = svgHtml;
}

function renderActivityChart(activity) {
    const container = document.getElementById("activityChartContainer");
    if (!container) return;

    if (!activity || activity.length === 0) {
        container.innerHTML = `
            <div style="width: 100%; display: flex; align-items: center; justify-content: center; height: 100%; color: #64748b; font-size: 12px;">
                No daily activity records logged yet
            </div>
        `;
        return;
    }

    const maxCount = Math.max(...activity.map(a => (a.total !== undefined ? a.total : a.count) || 0), 1);
    const maxBarHeight = 110;

    const html = activity.map(item => {
        const totalDay = item.total !== undefined ? item.total : (item.count || 0);
        const barHeight = Math.max(Math.round((totalDay / maxCount) * maxBarHeight), 8);

        const realPct = totalDay > 0 ? ((item.real || 0) / totalDay) * 100 : 0;
        const fakePct = totalDay > 0 ? ((item.fake || 0) / totalDay) * 100 : 0;
        const uncertainPct = totalDay > 0 ? ((item.uncertain || 0) / totalDay) * 100 : 0;

        let dateLabel = item.date;
        try {
            const parts = item.date.split("-");
            if (parts.length === 3) {
                const d = new Date(parts[0], parts[1] - 1, parts[2]);
                dateLabel = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
            }
        } catch (_) {}

        return `
            <div class="timeline-col" title="${item.date}: ${totalDay} predictions (${item.real || 0} Real, ${item.fake || 0} Fake, ${item.uncertain || 0} Uncertain)">
                <span class="timeline-count">${totalDay}</span>
                <div class="timeline-bar-stack" style="height: ${barHeight}px;">
                    ${realPct > 0 ? `<div class="stack-slice real" style="height: ${realPct}%;"></div>` : ""}
                    ${fakePct > 0 ? `<div class="stack-slice fake" style="height: ${fakePct}%;"></div>` : ""}
                    ${uncertainPct > 0 ? `<div class="stack-slice uncertain" style="height: ${uncertainPct}%;"></div>` : ""}
                </div>
                <span class="timeline-date">${dateLabel}</span>
            </div>
        `;
    }).join("");

    container.innerHTML = html;
}

function renderRecentActivity(recent) {
    const list = document.getElementById("recentActivityList");
    if (!list) return;

    if (!recent || recent.length === 0) {
        list.innerHTML = `
            <div style="color: #64748b; font-size: 12.5px; padding: 18px; text-align: center;">
                No recent predictions found. Check an article on the Analyze tab to get started.
            </div>
        `;
        return;
    }

    const html = recent.map(item => {
        let badgeClass = "uncertain";
        if (item.prediction.includes("REAL")) badgeClass = "real";
        else if (item.prediction.includes("FAKE")) badgeClass = "fake";

        const formattedDate = formatTimestamp(item.timestamp);
        const rawText = item.full_text || item.article_text || item.preview || "";
        const snippet = escapeHtml(item.preview || (rawText.length > 110 ? rawText.slice(0, 110) + "..." : rawText));
        const itemId = Number(item.id) || Date.now();
        historyTextStore.set(itemId, rawText);

        return `
            <div class="recent-item">
                <div class="recent-item-main">
                    <div class="recent-item-header">
                        <span class="history-badge ${badgeClass}">${escapeHtml(item.prediction)}</span>
                        <span class="history-conf">${item.confidence}% conf</span>
                        <span class="recent-time">🕒 ${formattedDate}</span>
                    </div>
                    <div class="recent-preview" title="${snippet}">
                        ${snippet}
                    </div>
                </div>
                <button type="button" class="btn-sm-action" onclick="inspectHistoryById(${itemId})">
                    Inspect ↗
                </button>
            </div>
        `;
    }).join("");

    list.innerHTML = html;
}

/* ==========================================
   Prediction History
   ========================================== */
let currentHistoryPage = 1;
let currentHistoryFilter = "ALL";
let currentHistorySearch = "";
let searchDebounceTimer = null;

function loadHistory(page = 1) {
    currentHistoryPage = page;

    const loadingElem = document.getElementById("historyLoading");
    const listElem = document.getElementById("historyList");
    const emptyElem = document.getElementById("historyEmpty");

    if (loadingElem) loadingElem.classList.remove("hidden");
    if (emptyElem) emptyElem.classList.add("hidden");

    const params = new URLSearchParams({
        page: page,
        limit: 10,
        filter: currentHistoryFilter,
        search: currentHistorySearch
    });

    fetch(`/api/history?${params.toString()}`)
        .then(res => res.json())
        .then(data => {
            if (loadingElem) loadingElem.classList.add("hidden");
            renderHistoryList(data);
        })
        .catch(err => {
            console.error("Failed to load history:", err);
            if (loadingElem) loadingElem.classList.add("hidden");
            if (listElem) {
                listElem.innerHTML = `<div style="color: #ef4444; padding: 20px; text-align: center;">Failed to load prediction history. Please check connection.</div>`;
            }
        });
}

function renderHistoryList(data) {
    const listElem = document.getElementById("historyList");
    const emptyElem = document.getElementById("historyEmpty");
    const emptyTextElem = document.getElementById("historyEmptyText");
    const paginationInfo = document.getElementById("paginationInfo");
    const btnPrev = document.getElementById("btnPrevPage");
    const btnNext = document.getElementById("btnNextPage");
    const pageIndicator = document.getElementById("pageIndicator");

    const total = data.total || 0;
    const page = data.page || 1;
    const totalPages = data.totalPages || data.total_pages || 1;
    const records = data.items || data.records || [];

    // Update tab badge count
    const badge = document.getElementById("historyCountBadge");
    if (badge && currentHistoryFilter === "ALL" && !currentHistorySearch) {
        badge.innerText = total.toLocaleString();
    }

    if (records.length === 0) {
        if (listElem) listElem.innerHTML = "";
        if (emptyElem) emptyElem.classList.remove("hidden");
        if (emptyTextElem) {
            if (currentHistorySearch || currentHistoryFilter !== "ALL") {
                emptyTextElem.innerText = "No predictions match your search query or filter criteria.";
            } else {
                emptyTextElem.innerText = "No predictions recorded yet. Run a verification on the Analyze tab to save records.";
            }
        }
        if (paginationInfo) paginationInfo.innerText = "Showing 0 of 0 records";
        if (btnPrev) btnPrev.disabled = true;
        if (btnNext) btnNext.disabled = true;
        if (pageIndicator) pageIndicator.innerText = "Page 1 of 1";
        return;
    }

    if (emptyElem) emptyElem.classList.add("hidden");

    // Render record cards
    if (listElem) {
        listElem.innerHTML = records.map(item => {
            let badgeClass = "uncertain";
            if (item.prediction.includes("REAL")) badgeClass = "real";
            else if (item.prediction.includes("FAKE")) badgeClass = "fake";

            const formattedDate = formatTimestamp(item.timestamp);
            const rawText = item.full_text || item.article_text || item.preview || "";
            const safeText = escapeHtml(rawText);
            const itemId = Number(item.id) || Date.now();
            historyTextStore.set(itemId, rawText);

            return `
                <div class="history-card-item" id="historyCard-${item.id}">
                    <div class="history-card-header">
                        <div class="history-meta-wrap">
                            <span class="history-badge ${badgeClass}">${escapeHtml(item.prediction)}</span>
                            <span class="history-conf">${item.confidence}% confidence</span>
                            <span class="history-version">${escapeHtml(item.model_version || "v2.1.0")}</span>
                        </div>
                        <span class="history-timestamp">🕒 ${formattedDate}</span>
                    </div>

                    <p class="history-preview-text">${safeText}</p>

                    <div class="history-actions-row">
                        <button type="button" class="btn-sm-action" onclick="inspectHistoryById(${itemId})">
                            🔍 Inspect in Detector
                        </button>
                        <button type="button" class="btn-sm-action delete" onclick="deleteHistoryItem(${item.id})" title="Delete record">
                            🗑️ Delete
                        </button>
                    </div>
                </div>
            `;
        }).join("");
    }

    // Pagination info & state
    const limit = data.limit || 10;
    const startIdx = (page - 1) * limit + 1;
    const endIdx = Math.min(page * limit, total);

    if (paginationInfo) {
        paginationInfo.innerText = `Showing ${startIdx}–${endIdx} of ${total} records`;
    }
    if (pageIndicator) {
        pageIndicator.innerText = `Page ${page} of ${totalPages}`;
    }
    if (btnPrev) btnPrev.disabled = page <= 1;
    if (btnNext) btnNext.disabled = page >= totalPages;
}

function onHistorySearchChange() {
    const input = document.getElementById("historySearch");
    const clearBtn = document.getElementById("btnClearSearch");
    const query = input ? input.value : "";

    if (clearBtn) {
        if (query.length > 0) clearBtn.classList.remove("hidden");
        else clearBtn.classList.add("hidden");
    }

    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
        currentHistorySearch = query.trim();
        loadHistory(1);
    }, 250);
}

function clearHistorySearch() {
    const input = document.getElementById("historySearch");
    const clearBtn = document.getElementById("btnClearSearch");
    if (input) input.value = "";
    if (clearBtn) clearBtn.classList.add("hidden");
    currentHistorySearch = "";
    loadHistory(1);
}

function setHistoryFilter(filter) {
    currentHistoryFilter = filter;

    const pills = document.querySelectorAll(".filter-tab");
    pills.forEach(pill => {
        if (pill.getAttribute("data-filter") === filter) {
            pill.classList.add("active");
        } else {
            pill.classList.remove("active");
        }
    });

    loadHistory(1);
}

function changeHistoryPage(delta) {
    const targetPage = currentHistoryPage + delta;
    if (targetPage >= 1) {
        loadHistory(targetPage);
    }
}

function loadIntoDetector(text) {
    const textarea = document.getElementById("newsText");
    if (textarea) {
        textarea.value = text;
        updateTextCounters();
    }
    switchTab("detector");
    window.scrollTo({ top: 0, behavior: "smooth" });
    checkNews();
}

function deleteHistoryItem(id) {
    fetch(`/api/history/${id}`, {
        method: "DELETE"
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "success" || data.success) {
            loadHistory(currentHistoryPage);
            updateHistoryBadge();
        }
    })
    .catch(err => {
        console.error("Failed to delete record:", err);
    });
}

function openClearModal() {
    const modal = document.getElementById("clearHistoryModal");
    if (modal) modal.classList.remove("hidden");
}

function closeClearModal() {
    const modal = document.getElementById("clearHistoryModal");
    if (modal) modal.classList.add("hidden");
}

function confirmClearHistory() {
    closeClearModal();
    fetch("/api/history", {
        method: "DELETE"
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "success" || data.success) {
            loadHistory(1);
            updateHistoryBadge();
            if (activeTab === "analytics") {
                loadAnalytics();
            }
        }
    })
    .catch(err => {
        console.error("Failed to clear history:", err);
    });
}

function updateHistoryBadge() {
    fetch("/api/history?page=1&limit=1")
        .then(res => res.json())
        .then(data => {
            const badge = document.getElementById("historyCountBadge");
            if (badge && typeof data.total === "number") {
                badge.innerText = data.total.toLocaleString();
            }
        })
        .catch(() => {});
}

function formatTimestamp(timestampStr) {
    if (!timestampStr) return "Just now";
    try {
        const date = new Date(timestampStr);
        if (isNaN(date.getTime())) return timestampStr;
        return date.toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit"
        });
    } catch (_) {
        return timestampStr;
    }
}

// Global initialization
document.addEventListener("DOMContentLoaded", () => {
    updateTextCounters();
    updateHistoryBadge();

    // Close modal on Escape key or backdrop click
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            closeClearModal();
        }
    });

    const modal = document.getElementById("clearHistoryModal");
    if (modal) {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) {
                closeClearModal();
            }
        });
    }
});
