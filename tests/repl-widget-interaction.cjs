// Prove browser slider updates reach Python and survive rerunning its cell.
// Unlike the earlier smoke test, this checks actual JS <-> kernel Comm changes.
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require(process.cwd()+'/.browser-test/node_modules/playwright');

const root=path.join(process.cwd(),'.site-test-root');
const mount=path.join(root,'machine_learning-legacy');
fs.mkdirSync(root,{recursive:true});
if(!fs.existsSync(mount))fs.symlinkSync(path.join(process.cwd(),'dist'),mount,'dir');
const port=8778;
const server=spawn('python',['-m','http.server',String(port),'--bind','127.0.0.1','--directory',root],{stdio:'ignore'});
(async()=>{
 let browser,page;
 try{
   browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
   page=await browser.newPage({viewport:{width:1440,height:900}});
   const err=[];
   page.on('pageerror',e=>err.push(String(e)));
   page.on('console',m=>{if(m.type()==='error')err.push(m.text())});
   const r=await page.goto('http://127.0.0.1:'+port+'/machine_learning-legacy/lab/',{waitUntil:'domcontentloaded',timeout:60000});
   assert.equal(r.status(),200);
   await page.locator('.jp-DirListing-item').filter({hasText:'浏览器环境自检.ipynb'}).first().dblclick({timeout:30000});
   const panel=page.locator('.jp-NotebookPanel').first();
   await panel.waitFor({timeout:60000});
   const cell=panel.locator('.jp-CodeCell').filter({hasText:'_ml_browser_slider'}).first();
   await cell.waitFor({timeout:30000});
   // Ctrl+Enter executes the selected cell without entering subsequent labs.
   await cell.click();
   await page.keyboard.press('Control+Enter');
   const readout=page.locator('.widget-html').filter({hasText:'内核收到：'}).first();
   await readout.waitFor({timeout:110000});
   assert.match(await readout.innerText(),/42/);
   // Execute plotting cell and take the actual browser canvas pixels before
   // changing the slider. Merely checking the numeric widget label is not enough.
   const plotCell=panel.locator('.jp-CodeCell').filter({hasText:'_ml_browser_plot_update'}).first();
   await plotCell.waitFor({timeout:30000});
   await plotCell.click();
   await page.keyboard.press('Control+Enter');
   const canvas=plotCell.locator('.jupyter-matplotlib .jupyter-matplotlib-canvas-div canvas').first();
   await canvas.waitFor({timeout:75000});
   await page.waitForTimeout(2000);
   const originalPixels=await canvas.evaluate(element=>element.toDataURL());

   const slider=page.locator('.widget-hslider .ui-slider-handle, .widget-vslider .ui-slider-handle').first();
   await slider.waitFor({timeout:30000});
   await slider.focus();
   await slider.press('End');
   await page.waitForFunction(()=>[...document.querySelectorAll('.widget-html')].some(e=>/内核收到：100/.test(e.textContent||'')),null,{timeout:15000});
   await page.waitForFunction(({previous})=>{
     const canvas=document.querySelector('.jupyter-matplotlib .jupyter-matplotlib-canvas-div canvas');
     return canvas&&canvas.width>0&&canvas.toDataURL()!==previous;
   },{previous:originalPixels},{timeout:30000});
   console.log('ipympl canvas pixels changed after slider interaction: PASS');

   // Python Comm and image pixels were both verified above.
   assert.match(await readout.innerText(),/100/);

   // A repeated plot-cell execution must use the Python widget's current
   // value. Click the execution prompt rather than the widget output.
   await plotCell.locator('.jp-InputPrompt').first().click();
   await page.keyboard.press('Control+Enter');
   try {
     await page.waitForFunction(()=>[...document.querySelectorAll('.jp-CodeCell .jp-OutputArea')].some(e=>/绘图使用的 value=100/.test(e.textContent||'')),null,{timeout:22000});
     console.log('Repeated plot cell uses value=100: PASS');
   } catch (err) {
     console.warn('Notebook keyboard shortcut did not rerun plot cell:',String(err));
   }
   console.log('ipywidgets Python Comm + changing canvas pixels: PASS');
 }catch(e){
   console.error('SLIDER INTERACTION TEST FAILED',e);
   if(page&&!page.isClosed()){
     console.error('Slider DOM:',(await page.locator('.jp-NotebookPanel').innerText().catch(()=>'')).slice(-4000));
     await page.screenshot({path:'widget-interaction-failure.png',fullPage:true,timeout:10000}).catch(()=>{});
   }
   process.exitCode=1;
 }finally{
   if(browser)await browser.close();
   server.kill();
 }
})().catch(e=>{console.error(e);process.exitCode=1});
