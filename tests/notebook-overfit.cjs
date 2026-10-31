// Reproduce the two historical ipympl canvas restore errors in a real Lab.
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require(process.cwd()+'/.browser-test/node_modules/playwright');
const root=path.join(process.cwd(),'.site-test-root');
const mount=path.join(root,'machine_learning-legacy');
fs.mkdirSync(root,{recursive:true});
if(!fs.existsSync(mount))fs.symlinkSync(path.join(process.cwd(),'dist'),mount,'dir');
const port=8777;
const server=spawn('python',['-m','http.server',String(port),'--bind','127.0.0.1','--directory',root],{stdio:'ignore'});

async function openAndRun(browser,file){
  const context=await browser.newContext({viewport:{width:1440,height:900}});
  const page=await context.newPage();
  const errors=[];
  page.on('console',msg=>{if(msg.type()==='error')errors.push(msg.text())});
  page.on('pageerror',e=>errors.push(String(e)));
  try {
    await page.goto('http://127.0.0.1:'+port+'/machine_learning-legacy/lab/',{waitUntil:'domcontentloaded',timeout:60000});
    await page.locator('.jp-DirListing-item').filter({hasText:'2. Logistic regression'}).first().dblclick({timeout:30000});
    await page.locator('.jp-DirListing-item').filter({hasText:file}).first().dblclick({timeout:30000});
    const panel=page.locator('.jp-NotebookPanel').first();
    await panel.waitFor({timeout:40000});
    await page.waitForFunction(()=>/Python|Pyodide/i.test(document.querySelector('.jp-NotebookPanel .jp-Toolbar')?.textContent||''),null,{timeout:55000});
    const names=await page.locator('.lm-MenuBar-item').allInnerTexts();
    const menu=names.findIndex(x=>/^(?:Run|运行)$/i.test(x.trim()));
    assert(menu>=0,'Missing Run menu');
    await page.locator('.lm-MenuBar-item').nth(menu).click();
    const options=await page.locator('.lm-Menu-itemLabel').allInnerTexts();
    const idx=options.findIndex(x=>/^(?:Run All Cells|Run All|运行所有单元格|运行全部单元格|全部运行|运行所有)$/i.test(x.trim()));
    assert(idx>=0,'Missing Run All command: '+options.join(','));
    await page.locator('.lm-Menu-itemLabel').nth(idx).click();
    const count=file.startsWith('8.')?2:10;
    await page.waitForFunction(count=>{
      const inputs=[...document.querySelectorAll('.jp-NotebookPanel .jp-CodeCell .jp-InputPrompt')].map(x=>x.textContent);
      return inputs.some(x=>new RegExp('\\['+count+'\\]').test(x)) || inputs.some(x=>/\[\d+\]/.test(x)) && document.querySelectorAll('.jp-OutputArea-output').length>=2;
    },count,{timeout:110000});
    // Allow widget-model initialization, deferred image decoding, and redraws.
    await page.waitForTimeout(6500);
    const broken=errors.filter(x=>/drawImage|broken.*state|Error setting state/i.test(x));
    assert.equal(broken.length,0,'Canvas initialization errors: '+broken.join('\n'));
    console.log('ipympl interactive notebook init PASS:',file,'all errors:',JSON.stringify(errors.slice(0,5)));
  }finally{await context.close()}
}
(async()=>{
 let browser;
 try {
  browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
  for(const file of ['8. Overfitting.ipynb','9. Regularized Cost and Gradient.ipynb'])await openAndRun(browser,file);
 }finally{if(browser)await browser.close();server.kill()}
})().catch(e=>{console.error(e);process.exitCode=1});
