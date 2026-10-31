// Execute every course notebook in a real JupyterLite Chromium session.
// Each notebook gets a clean browser context and its own JSON report/log.
// Errors are diagnostic only; never use process exit status to gate deployment.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require(process.cwd() + '/.browser-ci/node_modules/playwright');

const suite = process.env.NOTEBOOK_SUITE || '1. Linear Regression';
const base = (process.env.SITE_URL || 'https://scluzlep.github.io/machine_learning/').replace(/\/?$/, '/');
const root = path.join(process.cwd(), 'notebook-report');
const timeoutMs = Number(process.env.NOTEBOOK_TIMEOUT_MS || 150000);
const maxCellText = 200000;
const results = [];
let browser;
fs.mkdirSync(path.join(root, 'notebooks'), {recursive: true});
fs.mkdirSync(path.join(root, 'screenshots'), {recursive: true});
const timestamp = () => new Date().toISOString();
const errorText = err => err && err.stack || String(err);
const log = (tag, value) => console.log('[' + timestamp() + '] [' + tag + '] ' + String(value));
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const append = (file, object) => fs.appendFileSync(file, JSON.stringify({at:timestamp(), ...object}) + '\n');
const key = name => crypto.createHash('sha1').update(name).digest('hex').slice(0,12);
const text = value => Array.isArray(value) ? value.join('') : String(value || '');

function notebooksForSuite() {
  const rootPath = path.join(process.cwd(), 'machine_learning');
  if (suite === 'self-check') return ['浏览器环境自检.ipynb'];
  const dir = path.join(rootPath, suite);
  if (!fs.existsSync(dir)) throw Error('Notebook suite does not exist: ' + suite);
  const walk = (directory, result) => {
    for (const e of fs.readdirSync(directory,{withFileTypes:true})) {
      const abs = path.join(directory,e.name);
      if (e.isDirectory()) walk(abs,result);
      else if (e.name.endsWith('.ipynb')) result.push(path.relative(rootPath,abs).split(path.sep).join('/'));
    }
  };
  const files=[];
  walk(dir,files);
  return files.sort((a,b)=>a.localeCompare(b,'zh-CN',{numeric:true}));
}
function fromNotebook(file) {
  const data = JSON.parse(fs.readFileSync(path.join('machine_learning',file),'utf8'));
  const code = (data.cells || []).map((c,index) => ({
    cellIndex:index,
    source:text(c.source),
    type:c.cell_type
  })).filter(c=>c.type==='code');
  return {code, cellCount:(data.cells||[]).length};
}
function buildUrl(file) {
  return new URL('lab/tree/' + file.split('/').map(encodeURIComponent).join('/'),base).href;
}
async function collect(page,source,tracking,rel) {
  // JupyterLite's /api/contents/<filename> route is NOT a public HTTP API
  // on static GitHub Pages (404). Use JupyterLab File > Download instead:
  // it exports the entire current in-memory Notebook, including virtualized
  // cells, Python stdout, stderr, errors, plots and widget MIME data.
  let model;
  try {
    await page.keyboard.press('Control+s');
    await sleep(1000);
    const menus=page.locator('.lm-MenuBar-item');
    const menuNames=await menus.allInnerTexts();
    const fileIndex=menuNames.findIndex(x=>/^(?:File|文件)$/i.test(x.trim()));
    if(fileIndex<0)throw Error('File menu not found: '+JSON.stringify(menuNames));
    await menus.nth(fileIndex).click({timeout:12000});
    const labels=page.locator('.lm-Menu-itemLabel');
    const labelsText=await labels.allInnerTexts();
    const downloadIndex=labelsText.findIndex(x=>/^(?:Download|下载)$/i.test(x.trim()));
    if(downloadIndex<0)throw Error('Download command not found: '+JSON.stringify(labelsText));
    const requested=page.waitForEvent('download',{timeout:20000});
    await labels.nth(downloadIndex).click({timeout:12000});
    const download=await requested;
    const savePath=path.join(root,'executed-notebooks',key(rel)+'.ipynb');
    fs.mkdirSync(path.dirname(savePath),{recursive:true});
    await download.saveAs(savePath);
    model=JSON.parse(fs.readFileSync(savePath,'utf8'));
    log('INFO','Exported complete executed Notebook: '+rel);
  }catch(e) {
    log('WARN','Notebook download unavailable for '+rel+': '+errorText(e));
  }
  if(model && Array.isArray(model.cells)) {
    const code=model.cells.filter(c=>c.cell_type==='code');
    const cells=source.code.map((cell,index)=>{
      if(!cell.source.trim()) return {cellIndex:cell.cellIndex,codeIndex:index,
        source:'',status:'SKIP',reason:'Intentionally empty code cell',
        executionCount:null,outputs:[],hasError:false};
      const current=code[index];
      const outputs=(current?.outputs||[]).map(o=>{
        const data=o.data||{};
        const plain=data['text/plain'];
        const txt=o.text||plain||'';
        const value={
          type:o.output_type||'unknown',ename:o.ename||null,
          evalue:o.evalue||null,mimeTypes:Object.keys(data),
          text:text(txt).slice(0,maxCellText),
          traceback:text(o.traceback).slice(0,maxCellText)
        };
        if(data['image/png'])value.imageBytesApprox=Math.round(String(data['image/png']).length*0.75);
        return value;
      });
      const hasError=outputs.some(o=>o.type==='error'||o.ename||/Traceback \(most recent call last\)/.test(o.traceback));
      const executed=typeof current?.execution_count==='number';
      return {cellIndex:cell.cellIndex,codeIndex:index,source:cell.source.slice(0,100000),
        status:hasError?'FAIL':executed?'PASS':'NOT_EXECUTED',
        executionCount:current?.execution_count??null,outputs,hasError};
    });
    return {method:'JupyterLab File Download (whole executed Notebook)',cells,
      sourceCodeCells:source.code.length,modelCodeCells:code.length,
      pass:cells.filter(c=>c.status==='PASS').length,
      fail:cells.filter(c=>c.status==='FAIL').length,
      skippedEmpty:cells.filter(c=>c.status==='SKIP').length,
      unexecuted:cells.filter(c=>c.status==='NOT_EXECUTED').length};
  }
  // Explicitly report incomplete coverage rather than falsely claiming PASS.
  const dom=await page.evaluate(()=>[...document.querySelectorAll('.jp-NotebookPanel .jp-CodeCell')].map((c,index)=>({
    index,prompt:c.querySelector('.jp-InputPrompt')?.textContent||'',
    output:[...c.querySelectorAll('.jp-OutputArea-output')].map(x=>x.innerText||x.textContent||'')
  }))).catch(e=>[]);
  const cells=source.code.map((c,i)=>({
    cellIndex:c.cellIndex,codeIndex:i,source:c.source.slice(0,100000),
    status:'NOT_VERIFIED',outputs:[],executionCount:null
  }));
  return {method:'Notebook export unavailable, virtualized DOM fallback; NOT verified',
    cells,visibleCodeCells:dom,sourceCodeCells:source.code.length,
    pass:0,fail:0,unexecuted:cells.length};
}

async function openNotebook(page,rel) {
  // GitHub Pages returns HTTP 404 for /lab/tree/* URLs.
  // Open through the actual JupyterLite file browser instead.
  await page.goto(new URL('lab/',base).href,{waitUntil:'domcontentloaded',timeout:55000});
  const panel=page.locator('.jp-NotebookPanel').first();
  const segments=rel.split('/');
  for(const segment of segments) {
    await page.locator('.jp-DirListing-item').filter({hasText:segment}).first().dblclick({timeout:35000});
  }
  await panel.waitFor({timeout:55000});
  await page.waitForFunction(()=>{
    const toolbar=document.querySelector('.jp-NotebookPanel .jp-Toolbar');
    return !!toolbar && /Python|Pyodide/i.test(toolbar.textContent||'') && !/No Kernel/i.test(toolbar.textContent||'');
  },null,{timeout:60000});
  return panel;
}
async function installTracker(page) {
  await page.evaluate(() => {
    const panel=document.querySelector('.jp-NotebookPanel');
    if(!panel) throw Error('Missing notebook panel');
    const cells=[...panel.querySelectorAll('.jp-CodeCell')];
    const state={seen:[],seenBusy:[],events:[],changed:[],initial:cells.map(c=>c.querySelector('.jp-InputPrompt')?.textContent||'')};
    const visited=new Set(),busy=new Set(),changed=new Set();
    function scan() {
      cells.forEach((cell,index)=>{
        const prompt=(cell.querySelector('.jp-InputPrompt')?.textContent||'').trim();
        if(/\[\s*\*\s*\]/.test(prompt))busy.add(index);
        if(prompt!==state.initial[index])changed.add(index);
        if(busy.has(index)||changed.has(index))visited.add(index);
      });
      state.seen=[...visited];
      state.seenBusy=[...busy];
      state.changed=[...changed];
    }
    const observer=new MutationObserver(scan);
    observer.observe(panel,{subtree:true,childList:true,characterData:true,attributes:true});
    window.__mlNotebookTrack={state,observer};
    scan();
  });
}
async function clickRunAll(page) {
  // Do not rely on #jp-mainmenu-run: that ID is not present in Lite.
  // The localized Lumino menubar is visible in the actual JupyterLab UI.
  const bar=page.locator('.lm-MenuBar-item');
  const entries=await bar.allInnerTexts();
  const index=entries.findIndex(t=>/^(?:运行|Run)$/i.test(t.trim()));
  if(index<0)throw Error('Run menubar entry missing: '+JSON.stringify(entries));
  await bar.nth(index).click({timeout:15000});
  const labels=page.locator('.lm-Menu-itemLabel');
  const names=await labels.allInnerTexts();
  const wanted=/^(?:Run All Cells|Run All|运行所有单元格|运行全部单元格|全部运行|运行所有)$/i;
  const found=names.findIndex(t=>wanted.test(t.trim()));
  if(found<0)throw Error('Run all command missing: '+JSON.stringify(names));
  await labels.nth(found).click({timeout:15000});
  return names[found].trim();
}
async function recordNotebook(rel) {
  const source=fromNotebook(rel);
  const id=key(rel);
  const filepath=path.join(root,'notebooks',id+'.json');
  const logs=path.join(root,'notebooks',id+'.jsonl');
  const record={path:rel,suite,id,started:timestamp(),sourceCodeCells:source.code.length,
    totalCells:source.cellCount,status:'FAIL',errors:[],consoleErrors:[],httpFailures:[],progress:[]};
  if(rel.includes('切勿运行')) {
    record.status='SKIP';
    record.reason='Notebook explicitly marked 切勿运行';
    fs.writeFileSync(filepath,JSON.stringify(record,null,2));
    return record;
  }
  let context,page;
  const start=Date.now();
  try {
    context=await browser.newContext({viewport:{width:1440,height:900},acceptDownloads:true});
    page=await context.newPage();
    page.on('console',m=>{
      const e={kind:'console',type:m.type(),text:m.text(),location:m.location()};
      append(logs,e);
      if(m.type()==='error') record.consoleErrors.push(e);
    });
    page.on('pageerror',e=>{
      const item={kind:'pageerror',message:errorText(e)};
      append(logs,item);record.consoleErrors.push(item);
    });
    page.on('response',r=>{
      if(r.status()>=400) {
        const item={kind:'HTTP',status:r.status(),url:r.url()};
        record.httpFailures.push(item);append(logs,item);
      }
    });
    page.on('requestfailed',r=>{
      const item={kind:'requestfailed',url:r.url(),message:r.failure()?.errorText};
      record.httpFailures.push(item);append(logs,item);
    });
    page.on('worker',w=>append(logs,{kind:'worker',url:w.url()}));
    await openNotebook(page,rel);
    record.opened=timestamp();
    const actual=await page.locator('.jp-NotebookPanel .jp-CodeCell').count();
    // JupyterLab virtualizes long notebooks; only mounted cells appear in DOM.
    append(logs,{kind:'notebook-opened',mountedCodeCells:actual,sourceCodeCells:source.code.length});
    if(source.code.length===0) {
      record.status='PASS';
      record.reason='Markdown-only Notebook rendered';
    } else {
      await installTracker(page);
      record.runCommand=await clickRunAll(page);
      record.runStarted=timestamp();
      const end=Date.now()+timeoutMs;
      let finished=false, previous='', stable=0;
      while(Date.now()<end) {
        await sleep(1200);
        const snapshot=await page.evaluate(()=>{
          const obj=window.__mlNotebookTrack?.state;
          const cells=[...document.querySelectorAll('.jp-NotebookPanel .jp-CodeCell')];
          const active=cells.some(c=>/\[\s*\*\s*\]/.test(c.querySelector('.jp-InputPrompt')?.textContent||''));
          const kernelText=document.querySelector('.jp-NotebookPanel .jp-Toolbar')?.textContent||'';
          const kernelStatus=[...document.querySelectorAll('.jp-KernelStatus')].map(x=>x.textContent).join(' ');
          return {seen:obj?.seen||[],seenBusy:obj?.seenBusy||[],changed:obj?.changed||[],
            active,kernelText,kernelStatus,uiCodeCells:cells.length};
        });
        const compact=JSON.stringify({seen:snapshot.seen,active:snapshot.active,ui:snapshot.uiCodeCells});
        if(compact!==previous) {
          previous=compact;
          append(logs,{kind:'execution-progress',...snapshot});
          record.progress.push({at:timestamp(),...snapshot});
          log('PROGRESS',rel+' '+compact);
        }
        const idle=/空闲|Idle/i.test((snapshot.kernelStatus||'')+' '+(snapshot.kernelText||'')) && !/忙碌|Busy/i.test((snapshot.kernelStatus||'')+' '+(snapshot.kernelText||''));
        if(snapshot.seen.length>0 && !snapshot.active && idle) stable++;
        else stable=0;
        if(stable>=6){finished=true;break;}
      }
      const tracking=await page.evaluate(()=>window.__mlNotebookTrack?.state||null);
      record.execution=await collect(page,source,tracking,rel);
      record.finished=finished;
      if(!finished) record.errors.push('Timed out waiting for Notebook kernel to become idle after run-all ('+timeoutMs+'ms)');
      if(record.execution.modelCodeCells!==undefined && record.execution.modelCodeCells!==source.code.length)
        record.errors.push('Contents API code cell count mismatch');
      if(record.execution.fail) record.errors.push(record.execution.fail+' code cell(s) have Python error output');
      if(record.execution.unexecuted) record.errors.push(record.execution.unexecuted+' code cell(s) have no verified new execution result');
      if(record.consoleErrors.length) record.errors.push(record.consoleErrors.length+' browser console error(s)');
      if(record.httpFailures.length) record.errors.push(record.httpFailures.length+' failed HTTP request(s)');
      if(record.execution.cells) for(const cell of record.execution.cells) {
        append(logs,{kind:'code-cell-result',codeIndex:cell.codeIndex,status:cell.status,
          executionCount:cell.executionCount,outputTypes:(cell.outputs||[]).map(x=>x.type)});
      }
      record.status=record.errors.length?'FAIL':'PASS';
    }
  }catch(e) {
    record.errors.push(errorText(e));
    record.status='FAIL';
    if(page&&!page.isClosed()) {
      try {
        const tracking=await page.evaluate(()=>window.__mlNotebookTrack?.state||null);
        record.execution=await collect(page,source,tracking);
      }catch(_){}
    }
  }finally {
    record.durationSeconds=Math.round((Date.now()-start)/100)/10;
    record.finishedAt=timestamp();
    if(record.status!=='PASS' && page&&!page.isClosed()) {
      await page.screenshot({path:path.join(root,'screenshots',id+'.png'),fullPage:true,timeout:12000}).catch(()=>{});
    }
    if(context) await context.close().catch(()=>{});
    fs.writeFileSync(filepath,JSON.stringify(record,null,2));
  }
  return record;
}
function summary() {
  const counts={PASS:0,FAIL:0,SKIP:0};
  for(const r of results) counts[r.status]=(counts[r.status]||0)+1;
  const markdown=['# JupyterLite all-Notebook execution diagnostics','',
    'Suite: '+suite,'',
    '**'+results.length+' notebooks: '+counts.PASS+' PASS / '+counts.FAIL+' FAIL / '+counts.SKIP+' SKIP**',
    '',
    '| Notebook | Status | Code cells | Executed | Empty skipped | Python errors | Seconds |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ...results.map(r=>'| '+r.path.replace(/\|/g,'\\|')+' | '+r.status+' | '+
      r.sourceCodeCells+' | '+(r.execution?.pass||0)+' | '+(r.execution?.skippedEmpty||0)+' | '+(r.execution?.fail||0)+
      ' | '+(r.durationSeconds||0)+' |'),'',
    'Each notebook has its own JSON cell-output report and JSONL browser/network log. FAIL does not affect Action status.'
  ].join('\n')+'\n';
  fs.writeFileSync(path.join(root,'results.json'),JSON.stringify({suite,base,counts,notebooks:results},null,2));
  fs.writeFileSync(path.join(root,'summary.md'),markdown);
  if(process.env.GITHUB_STEP_SUMMARY)fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY,markdown);
  log('SUMMARY',counts.PASS+' PASS / '+counts.FAIL+' FAIL / '+counts.SKIP+' SKIP');
  console.log(markdown);
}
async function main() {
  const files=notebooksForSuite();
  log('INFO','Suite='+suite+'; notebooks='+files.length+'; site='+base);
  browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
  for(const file of files) {
    log('START',file);
    const r=await recordNotebook(file);
    results.push(r);
    log(r.status,file+'; code='+r.sourceCodeCells+'; executed='+(r.execution?.pass||0)+'; emptySkipped='+(r.execution?.skippedEmpty||0)+
      '; errors='+(r.execution?.fail||0)+'; elapsed='+(r.durationSeconds||0)+'s');
    if(r.errors?.length)r.errors.forEach(e=>log('ERROR',file+' '+e.slice(0,2500)));
    summary();
  }
}
main().catch(e=>{
  log('FAIL','Runner exception: '+errorText(e));
  results.push({path:'[runner]',status:'FAIL',sourceCodeCells:0,errors:[errorText(e)]});
}).finally(async()=>{
  if(browser)await browser.close().catch(()=>{});
  summary();
  process.exitCode=0;
});
