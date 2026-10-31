// Browser regression for C2_W3: verify an actual Notebook ipympl canvas,
// not the console's "Figure" text or an old PNG MIME fallback.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {chromium} = require(process.cwd() + '/.browser-test/node_modules/playwright');

const port = 8781;
const root = path.join(process.cwd(), '.site-test-root');
const mount = path.join(root, 'machine_learning-legacy');
fs.mkdirSync(root, {recursive:true});
if (!fs.existsSync(mount)) fs.symlinkSync(path.join(process.cwd(), 'dist'), mount, 'dir');
const server = spawn('python', ['-m','http.server',String(port),'--bind','127.0.0.1','--directory',root], {stdio:'ignore'});

(async()=>{
  let browser, page;
  const errors=[];
  try {
    browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
    page=await browser.newPage({viewport:{width:1440,height:900}});
    page.on('pageerror',e=>errors.push('PAGE '+String(e)));
    page.on('console',m=>{if(m.type()==='error')errors.push('CONSOLE '+m.text())});
    const url='http://127.0.0.1:'+port+'/machine_learning-legacy/lab/';
    const r=await page.goto(url,{waitUntil:'domcontentloaded',timeout:60000});
    assert.equal(r.status(),200);
    await page.locator('.jp-DirListing-item').filter({hasText:'6. Neural Network In Practice 1'}).first().dblclick({timeout:30000});
    await page.locator('.jp-DirListing-item').filter({hasText:'C2_W3_Assignment.ipynb'}).first().dblclick({timeout:30000});
    const panel=page.locator('.jp-NotebookPanel').first();
    await panel.waitFor({timeout:60000});
    await page.waitForFunction(()=>/Python|Pyodide/i.test(document.querySelector('.jp-NotebookPanel .jp-Toolbar')?.textContent||''),null,{timeout:60000});
    const cells=panel.locator('.jp-CodeCell');
    await page.waitForFunction(()=>document.querySelectorAll('.jp-NotebookPanel .jp-CodeCell').length>=4,null,{timeout:30000});
    assert.match(await cells.nth(0).innerText(),/%matplotlib widget/);
    const labels=['imports and widget backend','warning setup','generate training data','first interactive scatter plot'];
    for (let i=0;i<4;i++) {
      const cell=cells.nth(i);
      const prompt=cell.locator('.jp-InputPrompt').first();
      await prompt.scrollIntoViewIfNeeded();
      await prompt.click();
      await page.keyboard.press('Control+Enter');
      // Wait for real cell execution to finish. Before-run saved prompts may
      // already contain a number, so a numeric prompt alone is not evidence.
      await page.waitForTimeout(1200);
      await page.waitForFunction(()=>{
        const status=[...document.querySelectorAll('.jp-KernelStatus')].map(n=>n.textContent||'').join(' ');
        const busy=[...document.querySelectorAll('.jp-NotebookPanel .jp-InputPrompt')].some(n=>/\[\s*\*\s*\]/.test(n.textContent||''));
        return !busy && (/空闲|Idle/i.test(status) || !/忙碌|Busy/i.test(status));
      },null,{timeout:90000});
      const output=await cell.locator('.jp-OutputArea').innerText().catch(()=>'');
      if (/Traceback|ModuleNotFoundError|ImportError|Error while importing|Error executing/i.test(output)) {
        throw Error(labels[i]+' failed: '+output.slice(-2500));
      }
      console.log('C2W3 executed',i,labels[i],'output:',output.slice(0,350));
    }
    const plot=cells.nth(3);
    await plot.locator('.jupyter-matplotlib').first().waitFor({timeout:25000});
    const measure=()=>plot.evaluate(cell=>{
      const els=[...cell.querySelectorAll('.jupyter-matplotlib-canvas-div canvas')];
      return els.map(c=>{
        let pixels=0,unique=0;
        if(c.width>0&&c.height>0){
          try{
            const data=c.getContext('2d',{willReadFrequently:true}).getImageData(0,0,c.width,c.height).data;
            for(let i=0;i<data.length;i+=4*8){
              if(data[i+3]>0 && (data[i]<245||data[i+1]<245||data[i+2]<245))pixels++;
            }
          }catch(e){return {width:c.width,height:c.height,error:String(e)}}
        }
        return {width:c.width,height:c.height,inkPixelsSampled:pixels};
      });
    });
    await page.waitForFunction(()=>{
      const cell=[...document.querySelectorAll('.jp-NotebookPanel .jp-CodeCell')][3];
      if(!cell)return false;
      return [...cell.querySelectorAll('.jupyter-matplotlib-canvas-div canvas')].some(c=>{
        if(c.width<100||c.height<100)return false;
        try{
          const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
          let ink=0;
          for(let i=0;i<d.length;i+=4*8)if(d[i+3]>0&&(d[i]<245||d[i+1]<245||d[i+2]<245)&&++ink>50)return true;
        }catch(e){}
        return false;
      });
    },null,{timeout:28000});
    console.log('C2W3 NOTEBOOK LIVE IPYMPL DRAWING: PASS',JSON.stringify(await measure()));
    assert(!(await plot.innerText()).includes('Loading widget'),'Widget model must render');
  }catch(e){
    console.error('C2W3 NOTEBOOK LIVE IPYMPL DRAWING: FAIL',e);
    if(page&&!page.isClosed()){
      const area=page.locator('.jp-NotebookPanel .jp-CodeCell').nth(3);
      console.error('Figure output:',(await area.locator('.jp-OutputArea').innerText().catch(()=>'' )).slice(-2000));
      console.error('Canvas DOM:',(await area.locator('.jupyter-matplotlib').first().evaluate(e=>e.outerHTML.slice(0,1200)).catch(()=>'')));
      console.error('Browser errors:',JSON.stringify(errors.slice(-30)));
      await page.screenshot({path:'c2-widget-failure.png',fullPage:true,timeout:12000}).catch(()=>{});
    }
    process.exitCode=1;
  }finally{
    if(browser)await browser.close();
    server.kill();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
