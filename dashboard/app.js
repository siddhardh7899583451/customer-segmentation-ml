/**
 * Customer Segmentation & Behavioral Analytics Platform
 * Zero-Backend Vanilla JavaScript Client Engine
 * Safe DOM Manipulation, Pure Canvas Charting, and Safe CSV Export
 */

(function () {
  "use strict";

  // Application State
  const state = {
    customers: [],
    personas: [],
    isSynthetic: true,
    filteredCustomers: [],
    currentPage: 1,
    pageSize: 15,
    searchQuery: "",
  };

  // Color Palette
  const PALETTE = {
    vip: { bg: "#ede9fe", text: "#6d28d9", hex: "#8b5cf6" },
    loyal: { bg: "#d1fae5", text: "#047857", hex: "#10b981" },
    recent: { bg: "#dbeafe", text: "#1d4ed8", hex: "#2563eb" },
    atRisk: { bg: "#fee2e2", text: "#b91c1c", hex: "#ef4444" },
    dormant: { bg: "#f3f4f6", text: "#4b5563", hex: "#6b7280" },
  };

  // DOM Elements Cache
  const elements = {
    datasetStatusBadge: document.getElementById("dataset-status-badge"),
    btnLoadDemo: document.getElementById("btn-load-demo"),
    btnOpenUpload: document.getElementById("btn-open-upload"),
    btnExportCsv: document.getElementById("btn-export-csv"),
    kpiTotalCustomers: document.getElementById("kpi-total-customers"),
    kpiTotalRevenue: document.getElementById("kpi-total-revenue"),
    kpiAvgAov: document.getElementById("kpi-avg-aov"),
    kpiAvgRecency: document.getElementById("kpi-avg-recency"),
    personasContainer: document.getElementById("personas-container"),
    customerTableBody: document.getElementById("customer-table-body"),
    tableSearchInput: document.getElementById("table-search-input"),
    paginationInfo: document.getElementById("pagination-info"),
    btnPrevPage: document.getElementById("btn-prev-page"),
    btnNextPage: document.getElementById("btn-next-page"),
    uploadModal: document.getElementById("upload-modal"),
    btnCloseModal: document.getElementById("btn-close-modal"),
    csvFileInput: document.getElementById("csv-file-input"),
    dropzone: document.getElementById("dropzone"),
    uploadErrorMessage: document.getElementById("upload-error-message"),
    chartVolumeRevenue: document.getElementById("chart-volume-revenue"),
    chartScatterRfm: document.getElementById("chart-scatter-rfm"),
    tabBtns: document.querySelectorAll(".tab-btn"),
    tabPanes: document.querySelectorAll(".tab-pane"),
  };

  /**
   * Safe string escaping to prevent XSS.
   */
  function escapeText(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  /**
   * Format currency values.
   */
  function formatCurrency(num) {
    return "$" + Number(num || 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  /**
   * Format numbers with commas.
   */
  function formatNumber(num) {
    return Number(num || 0).toLocaleString("en-US");
  }

  /**
   * Prevent CSV Formula Injection during client export.
   * Accepts optional isNumeric flag to distinguish numeric fields from untrusted text fields.
   */
  function sanitizeForCsv(val, isNumeric = false) {
    if (val === undefined || val === null) return "";
    if (isNumeric) {
      return String(val);
    }
    let s = String(val);
    const trimmed = s.trimStart();
    if (trimmed.length > 0 && ["=", "+", "-", "@", "\t", "\r"].includes(trimmed[0])) {
      s = "'" + s;
    }
    return s;
  }

  /**
   * Properly escape CSV cell values (handles quotes, commas, and newlines).
   */
  function escapeCsvField(val, isNumeric = false) {
    const s = sanitizeForCsv(val, isNumeric);
    if (s.includes(",") || s.includes('"') || s.includes("\n") || s.includes("\r")) {
      return '"' + s.replace(/"/g, '""') + '"';
    }
    return s;
  }

  /**
   * Derive grounded customer personas from customer dataset.
   */
  function derivePersonas(customers) {
    if (!customers || customers.length === 0) return [];

    const totalCust = customers.length;
    const totalRev = customers.reduce((sum, c) => sum + (Number(c.monetary) || 0), 0);

    // Compute medians for relative grounding
    const sortedMon = [...customers].map(c => Number(c.monetary) || 0).sort((a, b) => a - b);
    const sortedRec = [...customers].map(c => Number(c.recency) || 0).sort((a, b) => a - b);
    const sortedFreq = [...customers].map(c => Number(c.frequency) || 0).sort((a, b) => a - b);

    const medMon = sortedMon[Math.floor(sortedMon.length * 0.5)] || 0;
    const medRec = sortedRec[Math.floor(sortedRec.length * 0.5)] || 0;
    const medFreq = sortedFreq[Math.floor(sortedFreq.length * 0.5)] || 1;

    // Segment Assignment Logic
    customers.forEach((c) => {
      const mon = Number(c.monetary) || 0;
      const rec = Number(c.recency) || 0;
      const freq = Number(c.frequency) || 1;

      // Top 10% spenders or high frequency + low recency
      if (mon >= sortedMon[Math.floor(sortedMon.length * 0.85)] && rec <= medRec) {
        c.segment_name = "Top-Tier VIP Champions";
        c.badge_class = "badge-vip";
        c.cluster_id = 0;
      } else if (rec > 90 && (mon > medMon || freq > medFreq)) {
        c.segment_name = "At-Risk / Lapsed Customers";
        c.badge_class = "badge-at-risk";
        c.cluster_id = 3;
      } else if (freq >= medFreq && rec <= medRec) {
        c.segment_name = "Loyal Repeat Regulars";
        c.badge_class = "badge-loyal";
        c.cluster_id = 1;
      } else if (rec <= 45 && freq <= 3) {
        c.segment_name = "Recent Starters & Promising";
        c.badge_class = "badge-recent";
        c.cluster_id = 2;
      } else {
        c.segment_name = "Low-Engagement / Casual Shoppers";
        c.badge_class = "badge-dormant";
        c.cluster_id = 4;
      }
    });

    // Aggregate stats per segment
    const segmentMap = {};
    customers.forEach((c) => {
      const seg = c.segment_name;
      if (!segmentMap[seg]) {
        segmentMap[seg] = {
          name: seg,
          badgeClass: c.badge_class,
          clusterId: c.cluster_id,
          customers: [],
        };
      }
      segmentMap[seg].customers.push(c);
    });

    const personas = Object.values(segmentMap).map((grp) => {
      const cnt = grp.customers.length;
      const pct = ((cnt / totalCust) * 100).toFixed(1);
      const segRev = grp.customers.reduce((sum, c) => sum + (Number(c.monetary) || 0), 0);
      const revPct = totalRev > 0 ? ((segRev / totalRev) * 100).toFixed(1) : 0;
      const avgMon = segRev / cnt;
      const avgRec = grp.customers.reduce((sum, c) => sum + (Number(c.recency) || 0), 0) / cnt;
      const avgFreq = grp.customers.reduce((sum, c) => sum + (Number(c.frequency) || 0), 0) / cnt;
      const avgAov = grp.customers.reduce((sum, c) => sum + (Number(c.avg_order_value) || (c.monetary / (c.frequency || 1))), 0) / cnt;

      let interpretation = "";
      let recommendations = [];

      switch (grp.name) {
        case "Top-Tier VIP Champions":
          interpretation = "Core VIP clientele who drive disproportionate revenue and demonstrate deep brand loyalty.";
          recommendations = [
            "Enroll in exclusive VIP concierge or tiered reward loyalty programs.",
            "Offer early access to new product releases and premium white-glove support.",
            "Avoid aggressive discount coupons; focus on appreciation and exclusive value-adds.",
          ];
          break;
        case "Loyal Repeat Regulars":
          interpretation = "Dependable repeat buyers with strong brand affinity who purchase regularly in manageable ticket sizes.";
          recommendations = [
            "Cross-sell complementary categories to increase basket size (AOV).",
            "Introduce threshold-based volume incentives (e.g., 'Spend $75, get free premium shipping').",
            "Deploy automated replenishment reminders based on product reorder cycles.",
          ];
          break;
        case "Recent Starters & Promising":
          interpretation = "Fresh customer acquisitions currently evaluating product quality and delivery experience.";
          recommendations = [
            "Deliver a targeted onboarding welcome sequence highlighting best-sellers.",
            "Provide a second-purchase incentive valid within 14 days to establish repeat behavior.",
            "Request feedback or reviews to solidify relationship early in the customer lifecycle.",
          ];
          break;
        case "At-Risk / Lapsed Customers":
          interpretation = "Former steady buyers showing acute signs of disengagement and churn risk.";
          recommendations = [
            "Execute personalized win-back email campaigns with tailored incentives.",
            "Distribute customer satisfaction surveys to diagnose reasons for lapse (pricing, shipping, quality).",
            "Highlight newly stocked items matching historical category preferences.",
          ];
          break;
        default:
          interpretation = "Price-sensitive or occasional buyers with low initial investment in the brand.";
          recommendations = [
            "Automate low-cost re-engagement newsletters during peak seasonal clearance events.",
            "Limit expensive paid ad retargeting; optimize for low-cost organic email touchpoints.",
            "Highlight low-ticket entry-point products to stimulate impulse transactions.",
          ];
      }

      return {
        name: grp.name,
        badgeClass: grp.badgeClass,
        clusterId: grp.clusterId,
        count: cnt,
        pct: pct,
        revPct: revPct,
        avgMon: avgMon,
        avgRec: avgRec.toFixed(1),
        avgFreq: avgFreq.toFixed(1),
        avgAov: avgAov,
        interpretation: interpretation,
        recommendations: recommendations,
      };
    });

    // Sort by revenue contribution descending
    personas.sort((a, b) => b.revPct - a.revPct);
    return personas;
  }

  /**
   * Render Top KPI metric cards.
   */
  function renderKpis() {
    const totalCust = state.customers.length;
    const totalRev = state.customers.reduce((sum, c) => sum + (Number(c.monetary) || 0), 0);
    const avgAov = totalCust > 0 ? totalRev / state.customers.reduce((sum, c) => sum + (Number(c.frequency) || 1), 0) : 0;
    const avgRec = totalCust > 0 ? (state.customers.reduce((sum, c) => sum + (Number(c.recency) || 0), 0) / totalCust).toFixed(1) : 0;

    elements.kpiTotalCustomers.textContent = formatNumber(totalCust);
    elements.kpiTotalRevenue.textContent = formatCurrency(totalRev);
    elements.kpiAvgAov.textContent = formatCurrency(avgAov);
    elements.kpiAvgRecency.textContent = `${avgRec} days`;

    // Dataset status badge
    if (state.isSynthetic) {
      elements.datasetStatusBadge.textContent = "SYNTHETIC DEMO DATA";
      elements.datasetStatusBadge.className = "badge badge-at-risk";
    } else {
      elements.datasetStatusBadge.textContent = "VERIFIED REAL DATA";
      elements.datasetStatusBadge.className = "badge badge-loyal";
    }
  }

  /**
   * Render Segment Persona Cards safely via DOM manipulation.
   */
  function renderPersonas() {
    elements.personasContainer.innerHTML = "";

    state.personas.forEach((p) => {
      const card = document.createElement("div");
      card.className = "persona-card";

      const header = document.createElement("div");
      header.className = "persona-header";

      const titleGroup = document.createElement("div");
      titleGroup.className = "persona-title-group";

      const title = document.createElement("h3");
      title.textContent = p.name;
      titleGroup.appendChild(title);

      const badge = document.createElement("span");
      badge.className = `badge ${p.badgeClass}`;
      badge.textContent = `${p.pct}% of Customers`;

      header.appendChild(titleGroup);
      header.appendChild(badge);
      card.appendChild(header);

      // Metrics Grid
      const metricsRow = document.createElement("div");
      metricsRow.className = "persona-metrics-row";

      const mData = [
        { label: "Count", val: formatNumber(p.count) },
        { label: "Revenue Share", val: `${p.revPct}%` },
        { label: "Avg Spend", val: formatCurrency(p.avgMon) },
        { label: "Avg Recency", val: `${p.avgRec}d` },
      ];

      mData.forEach((m) => {
        const item = document.createElement("div");
        item.className = "p-metric-item";
        const lbl = document.createElement("div");
        lbl.className = "p-label";
        lbl.textContent = m.label;
        const v = document.createElement("div");
        v.className = "p-val";
        v.textContent = m.val;
        item.appendChild(lbl);
        item.appendChild(v);
        metricsRow.appendChild(item);
      });
      card.appendChild(metricsRow);

      // Description & Interpretation
      const body = document.createElement("div");
      body.className = "persona-body";
      const interpP = document.createElement("p");
      interpP.textContent = p.interpretation;
      body.appendChild(interpP);
      card.appendChild(body);

      // Action recommendations box
      const actBox = document.createElement("div");
      actBox.className = "persona-actions-box";
      const actH4 = document.createElement("h4");
      actH4.textContent = "Recommended Marketing Actions:";
      actBox.appendChild(actH4);

      const ul = document.createElement("ul");
      p.recommendations.forEach((r) => {
        const li = document.createElement("li");
        li.textContent = r;
        ul.appendChild(li);
      });
      actBox.appendChild(ul);
      card.appendChild(actBox);

      elements.personasContainer.appendChild(card);
    });
  }

  /**
   * Pure Canvas Chart 1: Bar Chart of Volume % vs Revenue % by Segment.
   */
  function renderVolumeRevenueChart() {
    const canvas = elements.chartVolumeRevenue;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    if (state.personas.length === 0) return;

    const padding = { top: 30, right: 20, bottom: 65, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Y Axis Max
    let maxVal = 0;
    state.personas.forEach((p) => {
      maxVal = Math.max(maxVal, Number(p.pct), Number(p.revPct));
    });
    maxVal = Math.ceil((maxVal * 1.15) / 10) * 10 || 50;

    // Grid lines
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#64748b";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "right";

    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const yVal = (maxVal / yTicks) * i;
      const yPos = padding.top + chartH - (yVal / maxVal) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, yPos);
      ctx.lineTo(width - padding.right, yPos);
      ctx.stroke();
      ctx.fillText(`${yVal.toFixed(0)}%`, padding.left - 8, yPos + 3);
    }

    // Bars
    const groupW = chartW / state.personas.length;
    const barW = Math.min(28, (groupW - 20) / 2);

    state.personas.forEach((p, idx) => {
      const groupCenter = padding.left + groupW * idx + groupW / 2;

      // Bar 1: Customer Volume % (Blue)
      const h1 = (Number(p.pct) / maxVal) * chartH;
      const x1 = groupCenter - barW - 2;
      const y1 = padding.top + chartH - h1;
      ctx.fillStyle = "#3b82f6";
      ctx.fillRect(x1, y1, barW, h1);

      // Bar 2: Revenue Contribution % (Green)
      const h2 = (Number(p.revPct) / maxVal) * chartH;
      const x2 = groupCenter + 2;
      const y2 = padding.top + chartH - h2;
      ctx.fillStyle = "#10b981";
      ctx.fillRect(x2, y2, barW, h2);

      // X-axis label
      ctx.fillStyle = "#334155";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      const shortName = p.name.split(" ")[0] || `Cluster ${idx}`;
      ctx.fillText(shortName, groupCenter, padding.top + chartH + 18);
      ctx.fillText(`${p.pct}% / ${p.revPct}%`, groupCenter, padding.top + chartH + 32);
    });

    // Legend
    ctx.textAlign = "left";
    ctx.fillStyle = "#3b82f6";
    ctx.fillRect(width - 240, 10, 12, 12);
    ctx.fillStyle = "#334155";
    ctx.font = "11px sans-serif";
    ctx.fillText("Customer Volume %", width - 222, 20);

    ctx.fillStyle = "#10b981";
    ctx.fillRect(width - 115, 10, 12, 12);
    ctx.fillStyle = "#334155";
    ctx.fillText("Revenue %", width - 98, 20);
  }

  /**
   * Pure Canvas Chart 2: 2D Behavioral Scatter Plot (Recency vs Monetary).
   */
  function renderScatterChart() {
    const canvas = elements.chartScatterRfm;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    if (state.customers.length === 0) return;

    const padding = { top: 30, right: 25, bottom: 45, left: 60 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Log-transformed scales for natural customer visualization
    const maxRec = 365;
    const maxMon = Math.max(...state.customers.map(c => Number(c.monetary) || 0), 100);

    // Axes lines
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + chartH);
    ctx.lineTo(width - padding.right, padding.top + chartH);
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartH);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = "#64748b";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Recency (Days Elapsed Since Last Purchase)", padding.left + chartW / 2, height - 10);

    ctx.save();
    ctx.translate(16, padding.top + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Monetary Spend ($ Total)", 0, 0);
    ctx.restore();

    // Points
    const colorMap = {
      "Top-Tier VIP Champions": "#8b5cf6",
      "Loyal Repeat Regulars": "#10b981",
      "Recent Starters & Promising": "#2563eb",
      "At-Risk / Lapsed Customers": "#ef4444",
      "Low-Engagement / Casual Shoppers": "#64748b",
    };

    state.customers.forEach((c) => {
      const rec = Math.min(Number(c.recency) || 0, maxRec);
      const mon = Math.max(Number(c.monetary) || 0, 1);

      // Plot x as linear recency, y as log scale monetary
      const xPos = padding.left + (rec / maxRec) * chartW;
      const logY = Math.log10(mon + 1);
      const logMax = Math.log10(maxMon + 1);
      const yPos = padding.top + chartH - (logY / logMax) * chartH;

      ctx.beginPath();
      ctx.arc(xPos, yPos, 4, 0, Math.PI * 2);
      ctx.fillStyle = colorMap[c.segment_name] || "#3b82f6";
      ctx.globalAlpha = 0.65;
      ctx.fill();
      ctx.globalAlpha = 1.0;
    });
  }

  /**
   * Render Filtered & Paginated Customer Table safely.
   */
  function renderCustomerTable() {
    elements.customerTableBody.innerHTML = "";

    const filtered = state.filteredCustomers;
    const total = filtered.length;
    const totalPages = Math.ceil(total / state.pageSize) || 1;
    state.currentPage = Math.min(state.currentPage, totalPages);

    const startIdx = (state.currentPage - 1) * state.pageSize;
    const pageData = filtered.slice(startIdx, startIdx + state.pageSize);

    if (pageData.length === 0) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 6;
      td.style.textAlign = "center";
      td.style.padding = "24px";
      td.style.color = "#64748b";
      td.textContent = "No customers matched your search query.";
      tr.appendChild(td);
      elements.customerTableBody.appendChild(tr);
      elements.paginationInfo.textContent = "Showing 0 of 0 customers";
      elements.btnPrevPage.disabled = true;
      elements.btnNextPage.disabled = true;
      return;
    }

    pageData.forEach((c) => {
      const tr = document.createElement("tr");

      const tdId = document.createElement("td");
      tdId.style.fontWeight = "600";
      tdId.textContent = c.customer_id || `CUST_${c.CustomerID || "N/A"}`;
      tr.appendChild(tdId);

      const tdRec = document.createElement("td");
      tdRec.textContent = `${c.recency || 0} days`;
      tr.appendChild(tdRec);

      const tdFreq = document.createElement("td");
      tdFreq.textContent = `${c.frequency || 1} orders`;
      tr.appendChild(tdFreq);

      const tdMon = document.createElement("td");
      tdMon.textContent = formatCurrency(c.monetary);
      tr.appendChild(tdMon);

      const tdAov = document.createElement("td");
      tdAov.textContent = formatCurrency(c.avg_order_value || (c.monetary / (c.frequency || 1)));
      tr.appendChild(tdAov);

      const tdSeg = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = `badge ${c.badge_class || "badge-dormant"}`;
      badge.textContent = c.segment_name || "Unassigned";
      tdSeg.appendChild(badge);
      tr.appendChild(tdSeg);

      elements.customerTableBody.appendChild(tr);
    });

    elements.paginationInfo.textContent = `Showing ${startIdx + 1}–${Math.min(startIdx + state.pageSize, total)} of ${total} customers (Page ${state.currentPage} of ${totalPages})`;
    elements.btnPrevPage.disabled = state.currentPage <= 1;
    elements.btnNextPage.disabled = state.currentPage >= totalPages;
  }

  /**
   * Filter customer table on search input.
   */
  function handleSearch(query) {
    state.searchQuery = (query || "").trim().toLowerCase();
    state.currentPage = 1;

    if (!state.searchQuery) {
      state.filteredCustomers = [...state.customers];
    } else {
      state.filteredCustomers = state.customers.filter((c) => {
        const id = String(c.customer_id || c.CustomerID || "").toLowerCase();
        const seg = String(c.segment_name || "").toLowerCase();
        return id.includes(state.searchQuery) || seg.includes(state.searchQuery);
      });
    }
    renderCustomerTable();
  }

  /**
   * Update full UI with state.customers.
   */
  function refreshDashboard() {
    state.personas = derivePersonas(state.customers);
    state.filteredCustomers = [...state.customers];
    state.currentPage = 1;

    renderKpis();
    renderPersonas();
    renderVolumeRevenueChart();
    renderScatterChart();
    renderCustomerTable();
  }

  /**
   * Load Demo Dataset.
   */
  function loadDemoData() {
    if (window.DEMO_CUSTOMERS_DATA && Array.isArray(window.DEMO_CUSTOMERS_DATA)) {
      state.customers = JSON.parse(JSON.stringify(window.DEMO_CUSTOMERS_DATA));
      state.isSynthetic = true;
      refreshDashboard();
    } else {
      showError("Demo customer dataset could not be loaded from sample_demo_data.js");
    }
  }

  /**
   * Parse CSV content safely.
   */
  function parseCsv(text) {
    const lines = text.trim().split(/\r\n|\n/);
    if (lines.length < 2) throw new Error("CSV file contains no data rows.");

    const parseLine = (line) => {
      const result = [];
      let current = "";
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
          if (inQuotes && line[i + 1] === '"') {
            current += '"';
            i++;
          } else {
            inQuotes = !inQuotes;
          }
        } else if (char === "," && !inQuotes) {
          result.push(current.trim());
          current = "";
        } else {
          current += char;
        }
      }
      result.push(current.trim());
      return result;
    };

    const headers = parseLine(lines[0]).map(h => h.replace(/^["']|["']$/g, "").trim());
    const headerLower = headers.map(h => h.toLowerCase());

    const isRfm = headerLower.includes("recency") && headerLower.includes("frequency") && headerLower.includes("monetary");

    // Strict required-column sets (case-insensitive). A file must match one full schema.
    // Do not accept a raw transaction file merely because it contains a single related column.
    const has = (name) => headerLower.includes(name.toLowerCase());
    const transRequired = ["InvoiceNo", "Quantity", "UnitPrice", "CustomerID"];
    const missingTrans = transRequired.filter((c) => !has(c));
    const isTrans = missingTrans.length === 0;

    if (!isRfm && !isTrans) {
      if (missingTrans.length && missingTrans.length < transRequired.length) {
        throw new Error(
          "Incomplete raw transaction CSV. Missing required column(s): " +
            missingTrans.join(", ") +
            ". Required: InvoiceNo, Quantity, UnitPrice, CustomerID (plus InvoiceDate recommended)."
        );
      }
      throw new Error(
        "CSV missing required columns. Provide either RFM metrics (recency, frequency, monetary) " +
          "or a complete raw transaction schema (InvoiceNo, Quantity, UnitPrice, CustomerID)."
      );
    }

    if (lines.length <= 1) {
      throw new Error("CSV appears to contain only a header row (or is empty). No data rows found.");
    }

    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      const values = parseLine(lines[i]);
      const row = {};
      headers.forEach((h, idx) => {
        row[h] = values[idx] !== undefined ? values[idx] : "";
      });
      rows.push(row);
    }

    if (isRfm) {
      // Direct RFM customer records
      return rows.map((r, i) => ({
        customer_id: r.customer_id || r.CustomerID || r.id || `CUST_${i + 1}`,
        recency: Math.max(0, parseFloat(r.recency || r.Recency) || 0),
        frequency: Math.max(1, parseInt(r.frequency || r.Frequency, 10) || 1),
        monetary: Math.max(0, parseFloat(r.monetary || r.Monetary) || 0),
        avg_order_value: parseFloat(r.avg_order_value || r.AvgOrderValue) || 0,
      }));
    } else {
      // Transaction logs aggregation
      const custMap = {};
      const refDate = new Date();

      rows.forEach((r) => {
        const cid = r.CustomerID || r.customer_id;
        if (!cid) return; // Ignore guest checkouts
        const invNo = String(r.InvoiceNo || "");
        if (invNo.toUpperCase().startsWith("C")) return; // Ignore cancellation for monetary spend

        const qty = parseFloat(r.Quantity) || 0;
        const price = parseFloat(r.UnitPrice) || 0;
        if (qty <= 0 || price <= 0) return;

        const spend = qty * price;
        const invDate = r.InvoiceDate ? new Date(r.InvoiceDate) : refDate;

        if (!custMap[cid]) {
          custMap[cid] = {
            customer_id: String(cid),
            latestDate: invDate,
            invoices: new Set(),
            monetary: 0,
          };
        }
        custMap[cid].invoices.add(invNo);
        custMap[cid].monetary += spend;
        if (invDate > custMap[cid].latestDate) {
          custMap[cid].latestDate = invDate;
        }
      });

      return Object.values(custMap).map((c) => {
        const diffDays = Math.max(0, Math.floor((refDate - c.latestDate) / (1000 * 60 * 60 * 24)));
        const freq = Math.max(1, c.invoices.size);
        const mon = Math.round(c.monetary * 100) / 100;
        return {
          customer_id: c.customer_id,
          recency: diffDays,
          frequency: freq,
          monetary: mon,
          avg_order_value: Math.round((mon / freq) * 100) / 100,
        };
      });
    }
  }

  /**
   * Handle Uploaded File.
   */
  function handleFileUpload(file) {
    if (!file) return;
    if (!file.name.endsWith(".csv")) {
      showError("Only .csv files are supported. Please select a valid CSV file.");
      return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      try {
        const parsed = parseCsv(e.target.result);
        if (parsed.length === 0) {
          throw new Error("No valid customer records could be parsed from the provided CSV file.");
        }
        state.customers = parsed;
        state.isSynthetic = false; // User provided real dataset
        closeModal();
        refreshDashboard();
      } catch (err) {
        showError(err.message || "Failed to process CSV file.");
      }
    };
    reader.onerror = function () {
      showError("File reading failed. Please ensure file permissions are valid.");
    };
    reader.readAsText(file);
  }

  /**
   * Safe CSV Export.
   */
  function exportAnalysisCsv() {
    if (!state.customers || state.customers.length === 0) {
      alert("No customer data to export.");
      return;
    }

    const headers = ["customer_id", "recency", "frequency", "monetary", "avg_order_value", "segment_name"];
    const csvRows = [headers.join(",")];

    state.customers.forEach((c) => {
      const row = [
        escapeCsvField(c.customer_id || c.CustomerID, false),
        escapeCsvField(c.recency, true),
        escapeCsvField(c.frequency, true),
        escapeCsvField(c.monetary, true),
        escapeCsvField(c.avg_order_value || (c.monetary / (c.frequency || 1)).toFixed(2), true),
        escapeCsvField(c.segment_name, false),
      ];
      csvRows.push(row.join(","));
    });

    const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `customer_segmentation_analysis_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /**
   * Modal and UI Controls.
   */
  function openModal() {
    elements.uploadModal.classList.add("active");
    elements.uploadErrorMessage.style.display = "none";
  }

  function closeModal() {
    elements.uploadModal.classList.remove("active");
  }

  function showError(msg) {
    elements.uploadErrorMessage.textContent = msg;
    elements.uploadErrorMessage.style.display = "block";
  }

  function setupTabs() {
    elements.tabBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        elements.tabBtns.forEach(b => b.classList.remove("active"));
        elements.tabPanes.forEach(p => p.classList.remove("active"));

        btn.classList.add("active");
        const activePane = document.getElementById(targetTab);
        if (activePane) activePane.classList.add("active");

        if (targetTab === "tab-charts") {
          setTimeout(() => {
            renderVolumeRevenueChart();
            renderScatterChart();
          }, 50);
        }
      });
    });
  }

  // Event Listeners
  function initListeners() {
    elements.btnLoadDemo.addEventListener("click", loadDemoData);
    elements.btnOpenUpload.addEventListener("click", openModal);
    elements.btnCloseModal.addEventListener("click", closeModal);
    elements.btnExportCsv.addEventListener("click", exportAnalysisCsv);

    elements.tableSearchInput.addEventListener("input", (e) => {
      handleSearch(e.target.value);
    });

    elements.btnPrevPage.addEventListener("click", () => {
      if (state.currentPage > 1) {
        state.currentPage--;
        renderCustomerTable();
      }
    });

    elements.btnNextPage.addEventListener("click", () => {
      const totalPages = Math.ceil(state.filteredCustomers.length / state.pageSize) || 1;
      if (state.currentPage < totalPages) {
        state.currentPage++;
        renderCustomerTable();
      }
    });

    // File Drag and Drop
    elements.dropzone.addEventListener("click", () => elements.csvFileInput.click());
    elements.csvFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
      }
    });

    elements.dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      elements.dropzone.style.borderColor = "var(--primary)";
    });

    elements.dropzone.addEventListener("dragleave", () => {
      elements.dropzone.style.borderColor = "var(--slate-200)";
    });

    elements.dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      elements.dropzone.style.borderColor = "var(--slate-200)";
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    window.addEventListener("resize", () => {
      renderVolumeRevenueChart();
      renderScatterChart();
    });
  }

  // Initialization
  function init() {
    setupTabs();
    initListeners();
    loadDemoData();
  }

  // Launch on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
