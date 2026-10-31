const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.cwd() + '/.browser-test/node_modules/playwright');

const port = 8771;
const root = path.join(process.cwd(), '.site-test-root');
const mount = path.join(root, 'machine_learning-legacy');
fs.mkdirSync(root, { recursive: true });
if (!fs.existsSync(mount)) fs.symlinkSync(path.join(process.cwd(), 'dist'), mount, 'dir');
const server = spawn('python', ['-m', 'http.server', String(port), '--bind', '127.0.0.1', '--directory', root], {stdio: 'ignore'});

(async () => {
  let browser;
  let page;
  try {
    browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
    page = await browser.newPage();
    const url = `http://127.0.0.1:${port}/machine_learning-legacy/lab/`;
    await page.goto(url, {waitUntil: 'domcontentloaded', timeout: 60000});
    await page.locator('.jp-DirListing-item').filter({hasText: '1. Linear Regression'}).first().dblclick({timeout: 60000});
    await page.locator('.jp-DirListing-item').filter({hasText: '4. Gradient Descent for Linear Regression.ipynb'}).first().dblclick({timeout: 60000});

    const first = page.locator('.jp-NotebookPanel .jp-CodeCell').first();
    await first.waitFor({timeout: 60000});
    // CodeMirror virtualizes editor DOM; .cm-content may be empty until focus.
    // The notebook document still contains the actual original course cell.
    // JupyterLite lazily renders notebook cell editors. The exact original
    // cell is checked in tests/test-imports-bootstrap.py; don't require a
    // CodeMirror text node before it becomes visible.
    await page.waitForFunction(() => {
      const toolbar = document.querySelector('.jp-NotebookPanel .jp-Toolbar');
      return /Python|Pyodide/i.test(toolbar?.textContent || '');
    }, null, {timeout:60000});
    console.log('Running the original gradient-descent import cell, without editing it');
    await first.click();
    await page.keyboard.press('Control+Enter');

    await page.waitForFunction(() => {
      const cell = document.querySelector('.jp-NotebookPanel .jp-CodeCell');
      const prompt = cell?.querySelector('.jp-InputPrompt')?.textContent || '';
      return /\[\d+\]/.test(prompt);
    }, null, {timeout: 210000});

    const output = await first.locator('.jp-OutputArea').innerText().catch(() => '');
    const prompt = await first.locator('.jp-InputPrompt').innerText();
    console.log('Original helper import result', JSON.stringify({prompt, output:output.slice(0,1500)}));
    assert.doesNotMatch(output, /ModuleNotFoundError|ImportError|Traceback|Error while importing|pyodide.*error/i,
      'Original course import must succeed without a manual %pip install');
    console.log('Original gradient descent Notebook: nested ipywidgets dependency auto-installed, import PASS');
  } catch(e) {
    if(page) console.error('Notebook error detail:', (await page.locator('.jp-NotebookPanel').innerText().catch(()=> '')).slice(-3500));
    console.error(e);
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
