const puppeteer = require('puppeteer');
const path = require('path');
const { exec } = require('child_process');

(async () => {
  const dir = path.join(__dirname, 'screenshots');
  const framesDir = path.join(dir, 'frames');

  // Create frames directory
  require('fs').mkdirSync(framesDir, { recursive: true });

  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 2 });

  await page.goto('http://localhost:8000', { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise(r => setTimeout(r, 5000));

  // Capture a smooth scroll video as frames
  const totalHeight = await page.evaluate(() => document.body.scrollHeight);
  const viewportHeight = 900;
  const scrollStep = 4; // pixels per frame
  const fps = 30;
  let frame = 0;

  // Hold at top for 2 seconds
  await page.evaluate(() => window.scrollTo(0, 0));
  for (let i = 0; i < fps * 2; i++) {
    const padded = String(frame).padStart(5, '0');
    await page.screenshot({ path: path.join(framesDir, `frame_${padded}.png`) });
    frame++;
  }
  console.log(`Top hold: ${frame} frames`);

  // Smooth scroll down
  let scrollPos = 0;
  while (scrollPos < totalHeight - viewportHeight) {
    scrollPos += scrollStep;
    await page.evaluate((y) => window.scrollTo(0, y), scrollPos);
    const padded = String(frame).padStart(5, '0');
    await page.screenshot({ path: path.join(framesDir, `frame_${padded}.png`) });
    frame++;
    if (frame % 100 === 0) console.log(`Frame ${frame}, scroll ${scrollPos}/${totalHeight - viewportHeight}`);
  }

  // Hold at bottom for 2 seconds
  for (let i = 0; i < fps * 2; i++) {
    const padded = String(frame).padStart(5, '0');
    await page.screenshot({ path: path.join(framesDir, `frame_${padded}.png`) });
    frame++;
  }
  console.log(`Total frames: ${frame}`);

  await browser.close();

  // Use ffmpeg to combine frames into video
  const outputPath = path.join(dir, 'dashboard_demo.mp4');
  const ffmpegCmd = `ffmpeg -y -framerate ${fps} -i "${framesDir}/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p -preset slow -crf 18 -vf "scale=1400:900" "${outputPath}"`;

  console.log('Encoding video with ffmpeg...');
  exec(ffmpegCmd, (err, stdout, stderr) => {
    if (err) {
      console.error('ffmpeg error:', err.message);
      console.error(stderr);
    } else {
      console.log(`Video saved: ${outputPath}`);
      // Clean up frames
      exec(`rm -rf "${framesDir}"`, () => {
        console.log('Frames cleaned up');
      });
    }
  });
})();
