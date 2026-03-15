const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 2 });

  await page.goto('http://localhost:8000', { waitUntil: 'networkidle0', timeout: 30000 });
  // Wait extra for charts to render
  await new Promise(r => setTimeout(r, 5000));

  const dir = path.join(__dirname, 'screenshots');

  // 1. Full page screenshot
  await page.screenshot({ path: path.join(dir, 'dashboard_full.png'), fullPage: true });
  console.log('Saved: dashboard_full.png');

  // 2. Top section (Signal + Performance + Basket)
  await page.evaluate(() => window.scrollTo(0, 0));
  await new Promise(r => setTimeout(r, 300));
  await page.screenshot({ path: path.join(dir, 'dashboard_top.png') });
  console.log('Saved: dashboard_top.png');

  // 3. Chart section
  const chartEl = await page.$('#main-chart');
  if (chartEl) {
    await chartEl.screenshot({ path: path.join(dir, 'chart_brent_vs_s3.png') });
    console.log('Saved: chart_brent_vs_s3.png');
  }

  // 4. Probability table + Recent trades
  await page.evaluate(() => {
    const el = document.querySelector('.prob-table');
    if (el) el.closest('.panel').scrollIntoView({ block: 'start' });
  });
  await new Promise(r => setTimeout(r, 300));
  await page.screenshot({ path: path.join(dir, 'dashboard_tables.png') });
  console.log('Saved: dashboard_tables.png');

  // 5. Bottom section (Rolling charts + Model info)
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await new Promise(r => setTimeout(r, 300));
  await page.screenshot({ path: path.join(dir, 'dashboard_bottom.png') });
  console.log('Saved: dashboard_bottom.png');

  await browser.close();
  console.log('All screenshots captured!');
})();
