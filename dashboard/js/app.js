/* ============================================================
   ITO-S3 OIL TERMINAL - Dashboard Application
   ============================================================ */

(function () {
  'use strict';

  var REFRESH_MS = 60000;
  var state = {
    mainChart: null,
    betaChart: null,
    r2Chart: null
  };

  // ---- HELPERS ------------------------------------------

  function fmt(n, d) {
    if (n == null || isNaN(n)) return '--';
    return Number(n).toFixed(d);
  }

  function fmtPct(n, d) {
    if (n == null || isNaN(n)) return '--';
    return (n >= 0 ? '+' : '') + Number(n).toFixed(d || 2) + '%';
  }

  function fmtPrice(n) {
    if (n == null || isNaN(n)) return '--';
    return '$' + Number(n).toFixed(2);
  }

  function fmtSign(n, d) {
    if (n == null || isNaN(n)) return '--';
    return (n >= 0 ? '+' : '') + Number(n).toFixed(d);
  }

  function el(id) { return document.getElementById(id); }

  function setText(id, text) {
    var node = el(id);
    if (node) node.textContent = text;
  }

  function setClass(id, cls) {
    var node = el(id);
    if (node) node.className = cls;
  }

  // Convert parallel arrays to TradingView [{time, value}] format
  function zipChart(dates, values) {
    if (!dates || !values) return [];
    var out = [];
    for (var i = 0; i < dates.length && i < values.length; i++) {
      if (values[i] != null && isFinite(values[i])) {
        out.push({ time: dates[i], value: values[i] });
      }
    }
    return out;
  }

  // ---- CLOCK --------------------------------------------

  function updateClock() {
    var now = new Date();
    var p = function (n) { return String(n).padStart(2, '0'); };
    setText('clock',
      now.getUTCFullYear() + '-' + p(now.getUTCMonth() + 1) + '-' + p(now.getUTCDate())
      + '  ' + p(now.getUTCHours()) + ':' + p(now.getUTCMinutes()) + ':' + p(now.getUTCSeconds()) + ' UTC'
    );
  }

  // ---- CONNECTION STATUS --------------------------------

  function setConnected(flag) {
    var liveEl = el('status-live');
    var discEl = el('status-disconnected');
    if (liveEl) liveEl.style.display = flag ? 'flex' : 'none';
    if (discEl) discEl.style.display = flag ? 'none' : 'flex';
  }

  // ---- FETCH --------------------------------------------

  function fetchJSON(url) {
    return fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      });
  }

  // ---- RENDER: SIGNAL -----------------------------------

  function renderSignal(data) {
    if (!data) return;

    var dir = (data.signal || 'FLAT').toUpperCase();
    var predRet = data.predicted_ret != null ? data.predicted_ret * 100 : 0;

    var dirEl = el('signal-direction');
    if (dirEl) {
      dirEl.textContent = dir;
      dirEl.className = 'signal-display__direction';
      dirEl.classList.add(dir === 'LONG' ? 'long' : dir === 'SHORT' ? 'short' : 'flat');
    }

    var badge = el('signal-badge');
    if (badge) {
      badge.textContent = dir;
      badge.className = 'panel__badge';
      badge.classList.add(dir === 'LONG' ? 'panel__badge--green' : dir === 'SHORT' ? 'panel__badge--red' : 'panel__badge--amber');
    }

    setText('signal-pred-ret', fmtSign(predRet, 3) + '%');
    var prEl = el('signal-pred-ret');
    if (prEl) prEl.className = 'value ' + (predRet >= 0 ? 'positive' : 'negative');

    setText('signal-implied', fmtPrice(data.implied_brent));
    setText('signal-conf', fmtPrice(data.confidence_low) + ' - ' + fmtPrice(data.confidence_high));
    setText('signal-beta', fmt(data.rolling_beta, 5));
    setText('signal-r2', fmt(data.rolling_r2, 4));
    setText('signal-brent', fmtPrice(data.current_brent));
  }

  // ---- RENDER: STRATEGY ---------------------------------

  function renderStrategy(data) {
    if (!data) return;

    var cumRet = data.cumulative_return != null ? data.cumulative_return * 100 : 0;
    var maxDD = data.max_drawdown != null ? data.max_drawdown * 100 : 0;
    var accuracy = data.directional_accuracy != null ? data.directional_accuracy * 100 : 0;
    var nTrades = data.n_trades != null ? data.n_trades : 0;

    setText('strat-sharpe', fmt(data.sharpe, 2));
    setText('strat-cumulative', fmtSign(cumRet, 1) + '%');
    setText('strat-maxdd', fmt(maxDD, 1) + '%');
    setText('strat-calmar', fmt(data.calmar, 2));
    setText('strat-accuracy', fmt(accuracy, 0) + '%');
    setText('strat-trades', nTrades);
    var winRate = data.win_rate != null ? data.win_rate * 100 : null;
    var avgWin = data.avg_win != null ? data.avg_win * 100 : null;
    var avgLoss = data.avg_loss != null ? data.avg_loss * 100 : null;
    var pf = data.profit_factor;

    setText('strat-winrate', winRate != null ? fmt(winRate, 1) + '%' : '--');
    setText('strat-avgwin', avgWin != null ? fmtSign(avgWin, 2) + '%' : '--');
    setText('strat-avgloss', avgLoss != null ? fmtSign(avgLoss, 2) + '%' : '--');
    setText('strat-pf', pf != null ? fmt(pf, 2) : '--');

    var wEl = el('strat-avgwin');  if (wEl) wEl.className = 'value positive';
    var lEl = el('strat-avgloss'); if (lEl) lEl.className = 'value negative';

    var cumEl = el('strat-cumulative');
    if (cumEl) cumEl.className = 'value ' + (cumRet >= 0 ? 'positive' : 'negative');
    setClass('strat-maxdd', 'value negative');
  }

  // ---- RENDER: BASKET -----------------------------------

  function renderBasket(level, s3vals) {
    if (!s3vals || s3vals.length === 0) return;

    var current = s3vals[s3vals.length - 1];
    var prev = s3vals.length > 1 ? s3vals[s3vals.length - 2] : current;
    var v7d = s3vals.length > 7 ? s3vals[s3vals.length - 8] : s3vals[0];
    var v30d = s3vals.length > 30 ? s3vals[s3vals.length - 31] : s3vals[0];
    var v1y = s3vals.length > 252 ? s3vals[s3vals.length - 253] : s3vals[0];

    var todayRet = prev > 0 ? ((current - prev) / prev * 100) : 0;
    var ret7d = v7d > 0 ? ((current - v7d) / v7d * 100) : 0;
    var ret30d = v30d > 0 ? ((current - v30d) / v30d * 100) : 0;
    var ret1y = v1y > 0 ? ((current - v1y) / v1y * 100) : 0;

    setText('basket-level', fmt(current, 2));

    setText('basket-return', fmtSign(todayRet, 1) + '%');
    var retEl = el('basket-return');
    if (retEl) retEl.className = 'value ' + (todayRet >= 0 ? 'positive' : 'negative');

    var statusEl = el('basket-status');
    if (statusEl) {
      statusEl.textContent = 'ALL-TIME HIGH';
      statusEl.className = 'status-pill status-pill--ath';
    }

    setText('basket-7d', fmtSign(ret7d, 1) + '%');
    setText('basket-30d', fmtSign(ret30d, 1) + '%');
    setText('basket-1y', fmtSign(ret1y, 1) + '%');

    var e7 = el('basket-7d');  if (e7) e7.className = 'value ' + (ret7d >= 0 ? 'positive' : 'negative');
    var e30 = el('basket-30d'); if (e30) e30.className = 'value ' + (ret30d >= 0 ? 'positive' : 'negative');
    var e1y = el('basket-1y');  if (e1y) e1y.className = 'value ' + (ret1y >= 0 ? 'positive' : 'negative');
  }

  // ---- RENDER: PROBABILITIES ----------------------------

  function renderProbabilities(data) {
    var tbody = el('prob-tbody');
    if (!tbody || !data) return;

    // API returns: { gaussian: {"85": 0.99, ...}, student_t: {"85": 0.99, ...}, multi_horizon: {"21": {...}}, ... }
    var gauss = data.gaussian || {};
    var studt = data.student_t || {};
    var h21 = (data.multi_horizon && data.multi_horizon['21']) || {};
    var strikes = Object.keys(gauss).sort(function (a, b) { return Number(a) - Number(b); });

    if (strikes.length === 0) return;

    var html = '';
    strikes.forEach(function (s) {
      var gPct = (gauss[s] || 0) * 100;
      var tPct = (studt[s] || 0) * 100;
      var hPct = (h21[s] || 0) * 100;
      html += '<tr>'
        + '<td class="num">$' + s + '</td>'
        + '<td class="num">' + fmt(gPct, 1) + '%</td>'
        + '<td class="num">' + fmt(tPct, 1) + '%</td>'
        + '<td class="num" style="color:var(--gold);">' + fmt(hPct, 1) + '%</td>'
        + '</tr>';
    });
    tbody.innerHTML = html;
  }

  // ---- RENDER: TRADES -----------------------------------

  function renderTrades(trades) {
    var tbody = el('trades-tbody');
    if (!tbody || !trades || !trades.length) return;

    var html = '';
    // Show most recent first
    var sorted = trades.slice().reverse();
    sorted.forEach(function (t) {
      var sigClass = t.signal === 'LONG' ? 'positive' : t.signal === 'SHORT' ? 'negative' : 'text-amber';
      var pnl = (t.strat_ret || 0) * 100;
      var brentRet = (t.brent_ret || 0) * 100;
      var s3Ret = (t.s3_ret || 0) * 100;
      var pnlClass = pnl >= 0 ? 'positive' : 'negative';

      html += '<tr>'
        + '<td>' + t.date + '</td>'
        + '<td class="num">' + fmtSign(brentRet, 2) + '%</td>'
        + '<td class="' + sigClass + '">' + t.signal + '</td>'
        + '<td class="num ' + pnlClass + '">' + fmtSign(pnl, 2) + '%</td>'
        + '<td class="num">' + fmtSign(s3Ret, 2) + '%</td>'
        + '</tr>';
    });
    tbody.innerHTML = html;
  }

  // ---- CHARTS -------------------------------------------

  var TV_THEME = {
    layout: {
      background: { type: 'solid', color: '#ffffff' },
      textColor: '#8a8f9e',
      fontFamily: "'IBM Plex Mono', 'SF Mono', monospace",
      fontSize: 10
    },
    grid: {
      vertLines: { color: '#f0f0f4' },
      horzLines: { color: '#f0f0f4' }
    },
    crosshair: { mode: 0 },
    timeScale: {
      borderColor: '#e2e4e8',
      timeVisible: false,
      secondsVisible: false
    },
    handleScroll: { mouseWheel: true, pressedMouseMove: true },
    handleScale: { mouseWheel: true, pinch: true }
  };

  function destroyChart(chartObj) {
    if (chartObj && chartObj.remove) {
      try { chartObj.remove(); } catch (e) { /* ignore */ }
    }
  }

  function createMainChart(brentData, s3Data) {
    var container = el('main-chart');
    if (!container || typeof LightweightCharts === 'undefined') return;
    container.innerHTML = '';

    destroyChart(state.mainChart);

    var chart = LightweightCharts.createChart(container, Object.assign({}, TV_THEME, {
      width: container.clientWidth,
      height: 380,
      rightPriceScale: { borderColor: '#e2e4e8', scaleMargins: { top: 0.1, bottom: 0.1 } }
    }));

    var brentSeries = chart.addLineSeries({
      color: '#4a6fa5', lineWidth: 2, priceScaleId: 'left', title: 'BRENT',
      lastValueVisible: true, priceLineVisible: true, priceLineColor: '#4a6fa533',
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 }
    });

    var s3Series = chart.addLineSeries({
      color: '#8c7038', lineWidth: 2, priceScaleId: 'right', title: 'ITO-S3',
      lastValueVisible: true, priceLineVisible: true, priceLineColor: '#8c703833',
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 }
    });

    chart.priceScale('left').applyOptions({
      borderColor: '#e2e4e8',
      scaleMargins: { top: 0.1, bottom: 0.1 }
    });

    if (brentData.length) brentSeries.setData(brentData);
    if (s3Data.length) s3Series.setData(s3Data);
    chart.timeScale().fitContent();

    new ResizeObserver(function () {
      chart.applyOptions({ width: container.clientWidth });
    }).observe(container);

    state.mainChart = chart;
  }

  function createSmallChart(containerId, data, color, title, precision) {
    var container = el(containerId);
    if (!container || typeof LightweightCharts === 'undefined' || !data || !data.length) return;
    container.innerHTML = '';

    // Destroy previous
    if (containerId === 'beta-chart') destroyChart(state.betaChart);
    if (containerId === 'r2-chart') destroyChart(state.r2Chart);

    var chart = LightweightCharts.createChart(container, Object.assign({}, TV_THEME, {
      width: container.clientWidth,
      height: 200,
      rightPriceScale: {
        borderColor: '#1e1e2e',
        scaleMargins: { top: 0.15, bottom: 0.15 }
      }
    }));

    var series = chart.addLineSeries({
      color: color, lineWidth: 1.5, title: title,
      lastValueVisible: true, priceLineVisible: false,
      priceFormat: {
        type: 'price',
        precision: precision || 4,
        minMove: Math.pow(10, -(precision || 4))
      }
    });

    series.setData(data);
    chart.timeScale().fitContent();

    new ResizeObserver(function () {
      chart.applyOptions({ width: container.clientWidth });
    }).observe(container);

    if (containerId === 'beta-chart') state.betaChart = chart;
    if (containerId === 'r2-chart') state.r2Chart = chart;
  }

  // ---- MASTER REFRESH -----------------------------------

  function refreshDashboard() {
    var endpoints = [
      fetchJSON('/api/data'),
      fetchJSON('/api/signal'),
      fetchJSON('/api/strategy'),
      fetchJSON('/api/probabilities'),
      fetchJSON('/api/rolling')
    ];

    Promise.all(endpoints.map(function (p) { return p.catch(function () { return null; }); }))
      .then(function (results) {
        var apiData     = results[0];
        var apiSignal   = results[1];
        var apiStrategy = results[2];
        var apiProb     = results[3];
        var apiRolling  = results[4];

        var anySuccess = results.some(function (r) { return r !== null; });
        setConnected(anySuccess);

        // --- Signal panel ---
        try {
          if (apiSignal) renderSignal(apiSignal);
        } catch (e) { console.warn('renderSignal:', e); }

        // --- Strategy panel ---
        try {
          if (apiStrategy) renderStrategy(apiStrategy);
        } catch (e) { console.warn('renderStrategy:', e); }

        // --- Basket panel ---
        try {
          if (apiData && apiData.s3_level) {
            renderBasket(null, apiData.s3_level);
          }
        } catch (e) { console.warn('renderBasket:', e); }

        // --- Probability table ---
        try {
          if (apiProb) renderProbabilities(apiProb);
        } catch (e) { console.warn('renderProbabilities:', e); }

        // --- Trades table ---
        try {
          if (apiStrategy && apiStrategy.recent_trades) {
            renderTrades(apiStrategy.recent_trades);
          }
        } catch (e) { console.warn('renderTrades:', e); }

        // --- Main chart ---
        try {
          if (apiData) {
            var brentChart = zipChart(apiData.dates, apiData.brent_price);
            var s3Chart    = zipChart(apiData.dates, apiData.s3_level);
            if (brentChart.length) createMainChart(brentChart, s3Chart);
          }
        } catch (e) { console.warn('createMainChart:', e); }

        // --- Rolling charts ---
        try {
          if (apiRolling) {
            var betaData = zipChart(apiRolling.dates, apiRolling.beta);
            var r2Data   = zipChart(apiRolling.dates, apiRolling.r2);
            createSmallChart('beta-chart', betaData, '#4a6fa5', 'BETA', 5);
            createSmallChart('r2-chart',   r2Data,   '#8c7038', 'R2',   4);
          }
        } catch (e) { console.warn('createSmallChart:', e); }
      })
      .catch(function (err) {
        console.error('Dashboard refresh failed:', err);
        setConnected(false);
      });
  }

  // ---- INIT ---------------------------------------------

  function init() {
    updateClock();
    setInterval(updateClock, 1000);

    // Wait for TradingView library then start
    var attempts = 0;
    (function waitForTV() {
      if (typeof LightweightCharts !== 'undefined' || attempts > 30) {
        refreshDashboard();
        setInterval(refreshDashboard, REFRESH_MS);
      } else {
        attempts++;
        setTimeout(waitForTV, 200);
      }
    })();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
