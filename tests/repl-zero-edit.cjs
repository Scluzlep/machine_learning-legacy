const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.cwd() + '/.browser-test/node_modules/playwright');

const port = 8766;
const root = path.join(process.cwd(), '.site-test-root');
const mount = path.join(root, 'machine_learning-legacy');
fs.mkdirSync(root, {recursive: true});
if (!fs.existsSync(mount)) fs.symlinkSync(path.join(process.cwd(), 'dist'), mount, 'dir');
const server = spawn('python', ['-m', 'http.server', String(port), '--bind', '127.0.0.1', '--directory', root], {stdio: 'ignore'});

(async () => {
  let browser;
  try {
    browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
    const page = await browser.newPage();
    const messages = [];
    page.on('console', x => {
      const message = `[${x.type()}] ${x.text()}`;
      if (x.type() === 'error' || x.type() === 'warning') {
        messages.push(message);
        console.error('JupyterLite console:', message.slice(0, 1000));
      }
    });
    page.on('pageerror', error => {
      messages.push(String(error));
      console.error('JupyterLite page exception:', String(error));
    });
    page.on('requestfailed', req => {
      console.error('JupyterLite request failed:', req.url(), req.failure()?.errorText);
    });
    page.on('response', res => {
      if (res.status() >= 400) console.error('JupyterLite HTTP', res.status(), res.url());
    });
    // The notebook code remains conventional Python. The kernel handles
    // tensorflow imports and the asynchronous TF.js implementation.
    const python = [
      'import tensorflow as tf',
      'from tensorflow.keras.models import Sequential',
      'from tensorflow.keras.layers import Dense',
      'model = Sequential([Dense(2, activation="sigmoid", input_shape=(2,)), Dense(1, activation="sigmoid")])',
      'model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))',
      'model.fit([[0,0],[1,0],[0,1],[1,1]], [[0],[1],[1],[1]], epochs=2)',
      'from sklearn.linear_model import Ridge',
      'ridge = Ridge(alpha=0.25, normalize=True, max_iter=10000)',
      'ridge.fit([[1,2],[2,3],[3,5],[4,6]], [2,3,4,5])',
      'assert len(ridge.predict([[5,7]])) == 1',
      'print("ML_LEGACY_RIDGE_OK")',
      'from scipy import ndimage, misc',
      'from PIL import Image',
      'import numpy as np, tempfile, pathlib',
      'with tempfile.TemporaryDirectory() as folder:',
      '    path = pathlib.Path(folder) / "sample.png"',
      '    Image.fromarray(np.zeros((5,7,3), dtype=np.uint8)).save(path)',
      '    assert ndimage.imread(path).shape == (5,7,3)',
      '    assert misc.imresize(ndimage.imread(path), (3,4)).shape == (3,4,3)',
      'print("ML_LEGACY_SCIPY_OK")',
      'print("ML_KERNEL_ZERO_EDIT_OK")'
    ].join('\n');
    const url = `http://127.0.0.1:${port}/machine_learning-legacy/repl/index.html?kernel=python&code=${encodeURIComponent(python)}`;
    const response = await page.goto(url, {waitUntil:'domcontentloaded', timeout:60000});
    assert.equal(response.status(), 200, 'JupyterLite REPL must be available');
    try {
      await page.waitForFunction(
        () => [...document.querySelectorAll('.jp-OutputArea-output')]
           .map(node => node.textContent).join('\\n').includes('ML_KERNEL_ZERO_EDIT_OK') &&
          ['ML_LEGACY_RIDGE_OK','ML_LEGACY_SCIPY_OK'].every(marker =>
            [...document.querySelectorAll('.jp-OutputArea-output')].some(node => node.textContent.includes(marker))),
        null, {timeout:90000}
      );
    } catch (error) {
      console.error('REPL body excerpt:', (await page.locator('body').innerText().catch(()=>'')).slice(-4000));
      console.error('REPL console messages:', JSON.stringify(messages.slice(-40)));
      throw error;
    }
    console.log('Original Python TensorFlow syntax executed successfully in custom JupyterLite kernel');
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
