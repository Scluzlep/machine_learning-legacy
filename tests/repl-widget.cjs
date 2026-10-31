const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.cwd() + '/.browser-test/node_modules/playwright');

const port = 8772;
const root = path.join(process.cwd(), '.site-test-root');
const mount = path.join(root, 'machine_learning-legacy');
fs.mkdirSync(root, {recursive: true});
if (!fs.existsSync(mount)) fs.symlinkSync(path.join(process.cwd(), 'dist'), mount, 'dir');
const server = spawn('python', ['-m', 'http.server', String(port),
    '--bind', '127.0.0.1', '--directory', root], {stdio:'ignore'});

(async () => {
  let browser;
  try {
    browser = await chromium.launch({headless: true, args:['--no-sandbox']});
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', err => errors.push(String(err)));
    page.on('console', msg => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    const python = [
      'import ipywidgets as widgets',
      'from IPython.display import display',
      'slider = widgets.IntSlider(min=0, max=10, value=3, description="兼容测试")',
      'display(slider)',
      'assert slider.value == 3',
      'print("ML_COMM_WIDGET_OK")',
      "%matplotlib widget",
      'import matplotlib.pyplot as plt',
      'import warnings',
      'from pathlib import Path',
      'from matplotlib import font_manager',
      'from matplotlib.ft2font import FT2Font',
      'fontfile = font_manager.findfont(font_manager.FontProperties(family="Source Han Sans SC"), fallback_to_default=False)',
      'assert Path(fontfile).name == "SourceHanSansSC-Regular.otf", fontfile',
      'assert all(ord(ch) in FT2Font(fontfile).get_charmap() for ch in "交互式测试机器学习中文简体")',
      'fig, ax = plt.subplots()',
      'ax.plot([1,2,3], [1,4,9])',
      'ax.set_title("交互式 Matplotlib 测试")',
      'with warnings.catch_warnings(record=True) as w:',
      '    warnings.simplefilter("always")',
      '    fig.canvas.draw()',
      '    assert not any("Glyph" in str(item.message) for item in w), w',
      'print("ML_CJK_FONT_OK", font_manager.FontProperties(fname=fontfile).get_name())',

      'fig.canvas.mpl_connect("button_press_event", lambda event: None)',
      'assert fig.canvas.__class__.__module__.startswith("ipympl.")',
      'print("ML_WIDGET_BACKEND_OK", plt.get_backend(), fig.canvas.__class__.__module__)',
      'plt.show()'
    ].join('\n');
    const url = `http://127.0.0.1:${port}/machine_learning-legacy/repl/index.html?kernel=python&code=${encodeURIComponent(python)}`;
    const response = await page.goto(url, {waitUntil:'domcontentloaded', timeout:60000});
    assert.equal(response.status(), 200);
    await page.waitForFunction(
      () => [...document.querySelectorAll('.jp-OutputArea-output')]
          .some(x => x.textContent.includes('ML_WIDGET_BACKEND_OK')),
      null, {timeout: 90000}
    );
    const outputs = await page.locator('.jp-OutputArea-output').allInnerTexts();
    assert.match(outputs.join('\n'), /ML_WIDGET_BACKEND_OK.*ipympl/i);
    assert.match(outputs.join('\n'), /ML_CJK_FONT_OK Source Han Sans SC/i);
    assert.doesNotMatch(outputs.join('\n'), /Glyph \\d+ .*missing from font/i);

    assert.doesNotMatch(outputs.join('\n'), /not a recognised.*backend|Traceback|ModuleNotFoundError/);
    console.log('Matplotlib interactive ipympl + Adobe Source Han Sans SC glyphs: PASS');
  } catch(error) {
    console.error('Widget smoke failed:', error);
    try {
      if (browser) {
        const pages = browser.contexts().flatMap(context => context.pages());
        for (const p of pages) {
          const outputs = await p.locator('.jp-OutputArea-output').allInnerTexts().catch(() => []);
          console.error('Widget smoke outputs:', outputs.join('\\n').slice(-7000));
        }
      }
    } catch(_ignored) {}
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(error => {console.error(error);process.exitCode = 1});
