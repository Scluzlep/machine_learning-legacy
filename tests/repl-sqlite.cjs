// Verify that Pyodide 0.23.4 sqlite3 uses the same-origin package, not CDN.
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.cwd() + '/.browser-test/node_modules/playwright');

const root=path.join(process.cwd(), '.site-test-root');
const mount=path.join(root,'machine_learning-legacy');
fs.mkdirSync(root,{recursive:true});
if(!fs.existsSync(mount))fs.symlinkSync(path.join(process.cwd(),'dist'),mount,'dir');
const port=8776;
const server=spawn('python',['-m','http.server',String(port),'--bind','127.0.0.1','--directory',root],{stdio:'ignore'});
(async()=>{
  let browser;
  try {
    browser=await chromium.launch({headless:true,args:['--no-sandbox']});
    const context=await browser.newContext();
    // Simulate an environment where external Pyodide CDNs are unreachable.
    const external=[];
    await context.route(/^https?:\/\//, async route=>{
      const url=route.request().url();
      if(url.startsWith('http://127.0.0.1:'+port+'/'))return route.continue();
      external.push(url);
      return route.abort('blockedbyclient');
    });
    const page=await context.newPage();
    const requests=[];
    page.on('request',r=>{if(/sqlite3/i.test(r.url()))requests.push(r.url())});
    const code=[
      'import sqlite3',
      'conn = sqlite3.connect(":memory:")',
      'assert conn.execute("select 6*7").fetchone() == (42,)',
      'print("ML_LOCAL_SQLITE3_OK", sqlite3.sqlite_version)'
    ].join('\n');
    const url='http://127.0.0.1:'+port+'/machine_learning-legacy/repl/index.html?kernel=python&code='+encodeURIComponent(code);
    const res=await page.goto(url,{waitUntil:'domcontentloaded',timeout:60000});
    assert.equal(res.status(),200);
    await page.waitForFunction(()=>[...document.querySelectorAll('.jp-OutputArea-output')].some(x=>x.textContent.includes('ML_LOCAL_SQLITE3_OK')),null,{timeout:100000});
    assert(requests.some(u=>u.includes('/machine_learning-legacy/pyodide/sqlite3-')), 'sqlite3 must fetch from local Pyodide directory: '+JSON.stringify(requests));
    assert.equal(external.length,0,'All requests should stay on the Pages origin');
    console.log('sqlite3 executes and loads from local Pyodide without CDN: PASS');
  }finally{if(browser)await browser.close();server.kill()}
})().catch(err=>{console.error(err);process.exitCode=1});
