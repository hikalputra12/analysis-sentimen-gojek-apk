/**
 * Gojek Sentiment Intelligence Lab - Interactive Client Application
 */

const NEGATION_WORDS = new Set([
    "tidak", "bukan", "jangan", "belum", "tanpa", "kurang",
    "tak", "ndak", "ga", "gak", "nggak", "gk", "gda", "kagak"
]);

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const reviewInput = document.getElementById("review-input");
    const charCounter = document.getElementById("char-counter");
    const clearBtn = document.getElementById("clear-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    const btnText = document.getElementById("btn-text");
    const modelSelect = document.getElementById("model-select");
    const presetContainer = document.getElementById("preset-container");
    const modelBadge = document.getElementById("model-badge");

    // Result Elements
    const emptyState = document.getElementById("empty-state");
    const resultContent = document.getElementById("result-content");
    const sentimentBanner = document.getElementById("sentiment-banner");
    const sentimentTitle = document.getElementById("sentiment-title");
    const sentimentIcon = document.getElementById("sentiment-icon");
    const confidenceVal = document.getElementById("confidence-val");
    const topicTag = document.getElementById("topic-tag");
    const tokensContainer = document.getElementById("tokens-container");

    const barPos = document.getElementById("bar-pos");
    const barNeu = document.getElementById("bar-neu");
    const barNeg = document.getElementById("bar-neg");
    const pctPos = document.getElementById("pct-pos");
    const pctNeu = document.getElementById("pct-neu");
    const pctNeg = document.getElementById("pct-neg");

    const historyBody = document.getElementById("history-body");
    const clearHistoryBtn = document.getElementById("clear-history-btn");

    let historyItems = [];

    // Character counter
    reviewInput.addEventListener("input", () => {
        const len = reviewInput.value.length;
        charCounter.textContent = `${len} karakter`;
    });

    clearBtn.addEventListener("click", () => {
        reviewInput.value = "";
        charCounter.textContent = "0 karakter";
        reviewInput.focus();
    });

    // Model Selector Change
    modelSelect.addEventListener("change", () => {
        const selectedText = modelSelect.options[modelSelect.selectedIndex].text.split(" ")[0];
        modelBadge.textContent = modelSelect.value === "random_forest" ? "Random Forest" : "SVM (RBF)";
    });

    // Load Presets
    async function loadPresets() {
        try {
            const res = await fetch("/api/samples");
            const samples = await res.json();
            presetContainer.innerHTML = "";

            samples.forEach(sample => {
                const chip = document.createElement("button");
                chip.className = "chip-btn";
                chip.innerHTML = `<span>${getCategoryIcon(sample.category)}</span> <span>${sample.category}</span>`;
                chip.title = sample.text;
                chip.addEventListener("click", () => {
                    reviewInput.value = sample.text;
                    reviewInput.dispatchEvent(new Event("input"));
                    analyzeSentiment();
                });
                presetContainer.appendChild(chip);
            });
        } catch (err) {
            console.error("Failed to load sample presets:", err);
        }
    }

    function getCategoryIcon(cat) {
        if (cat.includes("GoFood")) return "🍔";
        if (cat.includes("GoPay")) return "💳";
        if (cat.includes("GPS") || cat.includes("Jemput")) return "📍";
        if (cat.includes("Ongkir")) return "💰";
        return "⚡";
    }

    // Analyze Sentiment
    async function analyzeSentiment() {
        const text = reviewInput.value.trim();
        if (!text) {
            alert("Silakan masukkan teks ulasan terlebih dahulu.");
            reviewInput.focus();
            return;
        }

        const model = modelSelect.value;

        // UI Loading State
        analyzeBtn.disabled = true;
        btnText.textContent = "Menganalisis Sentimen NLP...";

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: jsonStringifySafe({ text, model })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || "Gagal memproses prediksi.");
            }

            const data = await response.json();
            renderResults(data);
            addToHistory(data);
        } catch (err) {
            console.error("Analysis error:", err);
            alert(`Terjadi kesalahan: ${err.message}`);
        } finally {
            analyzeBtn.disabled = false;
            btnText.textContent = "Analisis Sentimen Sekarang";
        }
    }

    function jsonStringifySafe(obj) {
        return JSON.stringify(obj);
    }

    // Render Results
    function renderResults(data) {
        emptyState.classList.add("hidden");
        resultContent.classList.remove("hidden");

        const sentiment = data.sentiment.toLowerCase();
        const confPct = (data.confidence * 100).toFixed(1);

        // Reset classes
        sentimentBanner.className = "sentiment-banner";
        if (sentiment === "positive") {
            sentimentBanner.classList.add("pos");
            sentimentTitle.textContent = "POSITIVE";
            sentimentIcon.textContent = "🟢";
        } else if (sentiment === "neutral") {
            sentimentBanner.classList.add("neu");
            sentimentTitle.textContent = "NEUTRAL";
            sentimentIcon.textContent = "🟡";
        } else {
            sentimentBanner.classList.add("neg");
            sentimentTitle.textContent = "NEGATIVE";
            sentimentIcon.textContent = "🔴";
        }

        confidenceVal.textContent = `${confPct}%`;
        topicTag.textContent = data.topic_cluster;
        modelBadge.textContent = data.model_used;

        // Probabilities Bars
        const probs = data.probabilities || {};
        const posPct = Math.round((probs.positive || 0) * 100);
        const neuPct = Math.round((probs.neutral || 0) * 100);
        const negPct = Math.round((probs.negative || 0) * 100);

        barPos.style.width = `${posPct}%`;
        pctPos.textContent = `${posPct}%`;

        barNeu.style.width = `${neuPct}%`;
        pctNeu.textContent = `${neuPct}%`;

        barNeg.style.width = `${negPct}%`;
        pctNeg.textContent = `${negPct}%`;

        // Render Tokens
        tokensContainer.innerHTML = "";
        if (data.tokens && data.tokens.length > 0) {
            data.tokens.forEach(tok => {
                const span = document.createElement("span");
                span.className = "token-badge";
                if (NEGATION_WORDS.has(tok.toLowerCase())) {
                    span.classList.add("negation");
                    span.title = "Kata Negasi Kritis (Preserved)";
                }
                span.textContent = tok;
                tokensContainer.appendChild(span);
            });
        } else {
            tokensContainer.innerHTML = '<span class="text-subtle" style="font-size:0.8rem">Tidak ada token setelah pembersihan.</span>';
        }

        // Smooth Scroll on mobile
        if (window.innerWidth < 920) {
            document.getElementById("result-container").scrollIntoView({ behavior: "smooth" });
        }
    }

    // History Table Management
    function addToHistory(data) {
        const timeNow = new Date().toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
        historyItems.unshift({
            time: timeNow,
            text: data.text,
            model: data.model_used,
            topic: data.topic_cluster,
            sentiment: data.sentiment,
            confidence: `${(data.confidence * 100).toFixed(1)}%`
        });

        renderHistory();
    }

    function renderHistory() {
        if (historyItems.length === 0) {
            historyBody.innerHTML = '<tr class="empty-row"><td colspan="6">Belum ada riwayat ulasan yang diuji.</td></tr>';
            return;
        }

        historyBody.innerHTML = "";
        historyItems.slice(0, 10).forEach(item => {
            const tr = document.createElement("tr");
            const sentimentCls = item.sentiment === "positive" ? "pos" : (item.sentiment === "neutral" ? "neu" : "neg");
            
            tr.innerHTML = `
                <td style="white-space:nowrap; font-family:monospace; color:var(--text-subtle); font-size:0.75rem">${item.time}</td>
                <td style="max-width:320px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</td>
                <td style="font-size:0.75rem; color:var(--text-muted)">${item.model}</td>
                <td><span style="font-size:0.75rem; color:#38BDF8">${item.topic}</span></td>
                <td><span class="table-tag ${sentimentCls}">${item.sentiment.toUpperCase()}</span></td>
                <td style="font-weight:700; font-family:var(--font-heading)">${item.confidence}</td>
            `;
            historyBody.appendChild(tr);
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    clearHistoryBtn.addEventListener("click", () => {
        historyItems = [];
        renderHistory();
    });

    analyzeBtn.addEventListener("click", analyzeSentiment);

    reviewInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            analyzeSentiment();
        }
    });

    // Initial load
    loadPresets();
});
