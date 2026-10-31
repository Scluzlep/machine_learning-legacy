const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.cwd() + '/.browser-test/node_modules/playwright');

const port = 8767;
const root = path.join(process.cwd(), '.site-test-root');
const mount = path.join(root, 'machine_learning-legacy');
fs.mkdirSync(root, {recursive: true});
if (!fs.existsSync(mount)) fs.symlinkSync(path.join(process.cwd(), 'dist'), mount, 'dir');
const server = spawn('python', ['-m', 'http.server', String(port), '--bind', '127.0.0.1', '--directory', root], {stdio: 'ignore'});

(async () => {
  let browser;
  let page;
  try {
    browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
    const origin = `http://127.0.0.1:${port}`;
    page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', msg => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    for (const rel of [
      '1. Linear Regression/2.Model_Representation.ipynb',
      '4. Neural Network_part_1/2. CoffeeRoasting_TF.ipynb',
    ]) {
      if (page) await page.close();
      page = await browser.newPage();
      const [lesson, notebook] = rel.split('/');
      const response = await page.goto(`${origin}/machine_learning-legacy/lab/`, {waitUntil: 'domcontentloaded', timeout: 60000});
      assert.equal(response.status(), 200, 'Expected JupyterLab index');
      await page.locator('.jp-DirListing-item').filter({hasText: lesson}).first().dblclick({timeout: 60000});
      await page.locator('.jp-DirListing-item').filter({hasText: notebook}).first().dblclick({timeout: 60000});
      await page.locator('.jp-NotebookPanel').first().waitFor({timeout: 55000});
      await page.waitForFunction(() => {
        const panel = document.querySelector('.jp-NotebookPanel');
        if (!panel) return false;
        const toolbar = panel.querySelector('.jp-Toolbar');
        const text = toolbar?.textContent || '';
        return /Python|Pyodide/i.test(text) && !/No Kernel/i.test(text);
      }, null, {timeout: 60000});
      const toolbarText = await page.locator('.jp-NotebookPanel .jp-Toolbar').first().innerText();
      console.log('Notebook kernel selected:', rel, JSON.stringify(toolbarText.slice(0, 400)));
      assert.match(toolbarText, /Python|Pyodide/i);
    }
    const resource = await page.request.get(`${origin}/machine_learning-legacy/files/1.%20Linear%20Regression/images/3d_ds.jpg`);
    assert.equal(resource.status(), 200, 'Original Notebook image should be served from the static files tree');
    assert.match(resource.headers()['content-type'] || '', /image/i);
    console.log('Notebook UI kernel and static course images: PASS');
  } catch (err) {
    if (page) console.error('Notebook page text:', (await page.locator('body').innerText().catch(() => '')).slice(0, 2200));
    console.error(err);
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(err => { console.error(err); process.exitCode = 1; });
