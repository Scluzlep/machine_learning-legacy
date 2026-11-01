// Browser-level smoke test for the customized Pyodide kernel's Borch bridge.
// Run against the local Pages build; force CPU to validate old Chromium / Win7.
// Notebook self-check also exercises GPU automatically where available.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { chromium } = require(process.cwd() + '/.browser-test/node_modules/playwright');

const sitePrefix = process.env.TORCH_SITE_PREFIX ||
  ((process.env.GITHUB_REPOSITORY || '').endsWith('/machine_learning-legacy') ?
    'machine_learning-legacy' : 'machine_learning');
const port = 8783;
const root = path.join(process.cwd(), '.site-test-root');
const mount = path.join(root, sitePrefix);
fs.mkdirSync(root, {recursive:true});
if(!fs.existsSync(mount)) fs.symlinkSync(path.join(process.cwd(), 'dist'), mount, 'dir');

const source = [
  'import torch',
  'import torch.nn as nn',
  'import torch.nn.functional as F',
  'from torch.optim import SGD',
  'from torch.utils.data import DataLoader',
  'import numpy as np',
  'assert torch.__borch_compat__',
  'assert torch.__borch_backend__ == "numpy", torch.__borch_backend__',
  'x=torch.tensor([1.,2.,3.],requires_grad=True)',
  '(x*x).sum().backward()',
  'np.testing.assert_allclose(x.grad.tolist(),[2.,4.,6.],rtol=1e-5)',
  'assert F.relu(torch.tensor([-1.,2.])).tolist()==[0.,2.]',
  'net=nn.Linear(1,1)',
  'opt=SGD(net.parameters(),lr=0.04)',
  'a=torch.tensor([[0.],[1.],[2.],[3.]])',
  'b=torch.tensor([[1.],[3.],[5.],[7.]])',
  'losses=[]',
  'for i in range(80):',
  '    opt.zero_grad()',
  '    loss=((net(a)-b)**2).mean()',
  '    loss.backward()',
  '    opt.step()',
  '    if i in (0,79): losses.append(float(loss.item()))',
  'assert np.isfinite(losses).all() and losses[-1]<losses[0],losses',
  'print("ML_TORCH_CPU_PASS",torch.__borch_backend__,losses,flush=True)'
].join('\n');
const server = spawn('python', ['-m','http.server',String(port),'--bind','127.0.0.1','--directory',root],{stdio:'ignore'});
(async()=> {
 let browser,page;
 try{
  browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-features=WebGPU']});
  page=await browser.newPage();
  const url='http://127.0.0.1:'+port+'/'+sitePrefix+'/repl/index.html?kernel=python&code='+encodeURIComponent(source);
  const res=await page.goto(url,{waitUntil:'domcontentloaded',timeout:60000});
  assert.equal(res.status(),200);
  try{
    await page.waitForFunction(()=>[...document.querySelectorAll('.jp-OutputArea-output')].some(n=>n.textContent.includes('ML_TORCH_CPU_PASS')),null,{timeout:210000});
  }catch(e){
    const outputs=(await page.locator('.jp-OutputArea-output').allInnerTexts().catch(()=>[])).join('\n');
    throw new Error('Torch bridge did not complete: '+outputs.slice(-6500)+'\n'+String(e));
  }
  const output=(await page.locator('.jp-OutputArea-output').allInnerTexts()).join('\n');
  assert.doesNotMatch(output,/Traceback|ModuleNotFoundError|ImportError|ChunkLoadError/);
  console.log('Borch torch compatibility CPU fallback browser smoke: PASS\n'+output.slice(-1000));
 }finally{
  if(browser)await browser.close();
  server.kill();
 }
})().catch(e=>{console.error(e);process.exitCode=1});
