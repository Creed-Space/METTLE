import puppeteer from 'puppeteer-core';
import {createServer} from 'node:http';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {spawn,spawnSync} from 'node:child_process';import {once} from 'node:events';import path from 'node:path';import {createHash} from 'node:crypto';
const stillsOnly=process.argv.includes('--stills-only');
const fullPaint=process.argv.includes('--full-paint');
const [scene='01-hook',seconds='12',dest='video/build/remaster/graphics/proof.mp4',still]=process.argv.slice(2).filter(a=>!a.startsWith('--'));const duration=Number(seconds),root=path.resolve(import.meta.dirname,'../../..');
let autoBatchPath=null,autoStart=null;
// A legacy per-chapter caller may already be running. Reuse verified outputs and
// consume the remaining chapters in one browser on its next child invocation.
if(!stillsOnly&&!process.argv.some(a=>a.startsWith('--batch='))&&path.dirname(path.resolve(dest))===path.join(root,'video/build/remaster/graphics/final')){
 const manifestPath=path.join(path.dirname(path.resolve(dest)),'manifest.json');const manifest=JSON.parse(await readFile(manifestPath,'utf8'));
 const hash=createHash('sha256').update(await readFile(path.join(import.meta.dirname,'scenes.html'))).digest('hex');if(hash!==manifest.source_sha256){const approved=JSON.parse(await readFile(path.join(path.dirname(path.resolve(dest)),'source-conditional-edits.json'),'utf8'));if(approved.base_sha256!==manifest.source_sha256||approved.allowed_sha256!==hash)throw Error('Final source fingerprint changed without scoped record');}const job=manifest.scenes.find(s=>s.id===scene);
 const existing=spawnSync('ffprobe',['-v','error','-show_entries','stream=width,height,r_frame_rate,nb_frames','-of','json',path.resolve(dest)],{encoding:'utf8'});
 if(existing.status===0){const stream=JSON.parse(existing.stdout).streams[0];if(stream.width===1920&&stream.height===1080&&stream.r_frame_rate==='30/1'&&Number(stream.nb_frames)===job.frames){console.log(`${scene}: existing final clip independently re-probed, no duplicate render`);process.exit(0);}}
 autoBatchPath=manifestPath;autoStart=scene;
}
const server=createServer(async(req,res)=>{try{const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname);const file=path.resolve(root,'.'+name);if(!file.startsWith(root+'/'))throw Error('outside root');const bytes=await readFile(file);res.setHeader('Content-Type',file.endsWith('.html')?'text/html':file.endsWith('.js')?'application/javascript':file.endsWith('.woff2')?'font/woff2':file.endsWith('.png')?'image/png':'application/octet-stream');res.end(bytes)}catch{res.writeHead(404);res.end()}});await new Promise(r=>server.listen(0,'127.0.0.1',r));const port=server.address().port;let browser,encoder;const errors=[];
try{browser=await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,defaultViewport:{width:1920,height:1080,deviceScaleFactor:1},args:['--disable-background-networking','--window-size=1920,1080']});const page=await browser.newPage();page.on('pageerror',e=>errors.push(String(e)));await page.setViewport({width:1920,height:1080,deviceScaleFactor:1});await page.goto(`http://127.0.0.1:${port}/video/remaster/graphics/scenes.html?scene=${scene}&duration=${duration}`,{waitUntil:'networkidle0'});await page.waitForFunction(()=>window.ready);await page.evaluate(()=>document.fonts.ready.then(()=>true));
const proofArg=process.argv.find(a=>a.startsWith('--paint-proof='));
if(proofArg){
 const dir=path.resolve(proofArg.slice(14));await mkdir(dir,{recursive:true});const fast={type:'jpeg',quality:94,captureBeyondViewport:false,optimizeForSpeed:true};const full={type:'png',captureBeyondViewport:true,optimizeForSpeed:false};
 await page.evaluate(()=>window.seek(12));await page.screenshot({...fast,path:path.join(dir,'direct12-fast.jpg')});await page.screenshot({...full,path:path.join(dir,'direct12-full.png')});
 const dom=await page.evaluate(()=>[...document.querySelectorAll('h1,.stage,.a,.a svg,.orbit')].map(el=>({tag:el.tagName,cls:el.className.baseVal??el.className,rect:el.getBoundingClientRect().toJSON(),style:getComputedStyle(el).cssText,transform:getComputedStyle(el).transform,overflow:getComputedStyle(el).overflow,clipPath:getComputedStyle(el).clipPath})));await writeFile(path.join(dir,'dom12.json'),JSON.stringify(dom,null,2));
 await page.goto(`http://127.0.0.1:${port}/video/remaster/graphics/scenes.html?scene=${scene}&duration=${duration}`,{waitUntil:'networkidle0'});await page.evaluate(()=>document.fonts.ready.then(()=>true));
 for(let frame=0;frame<=360;frame++){await page.evaluate(t=>window.seek(t),frame/30);const bytes=await page.screenshot(fast);if([90,150,240,360].includes(frame))await writeFile(path.join(dir,`sequence${frame/30}-fast.jpg`),bytes);if(frame%150===0)console.log(`Paint proof frame${frame}/360`);}
 await page.screenshot({...full,path:path.join(dir,'sequence12-full.png')});await page.screenshot({...fast,optimizeForSpeed:false,path:path.join(dir,'sequence12-jpeg-slow.jpg')});
 if(errors.length)throw Error(JSON.stringify(errors));console.log('Paint proof complete, no page errors');
}else if(stillsOnly){
 const ids=['01-hook','02-what','03-challenges','04-suites','05-questions','06-governance','07-controls','08-credentials','09-uses','10-close'];const directory=path.resolve('video/build/remaster/graphics/stills');await mkdir(directory,{recursive:true});const bounds=[];
 const only=process.argv.find(a=>a.startsWith('--only='))?.slice(7).split(',');
 for(const id of ids.filter(id=>!only||only.includes(id))){await page.goto(`http://127.0.0.1:${port}/video/remaster/graphics/scenes.html?scene=${id}&duration=30`,{waitUntil:'networkidle0'});await page.waitForFunction(()=>window.ready);await page.evaluate(()=>document.fonts.ready.then(()=>true));
 const fit=await page.evaluate(()=>{const failures=[];for(const group of document.querySelectorAll('g.reveal')){const rect=group.querySelector('rect');if(!rect)continue;const r=rect.getBBox();for(const t of group.querySelectorAll('text')){const b=t.getBBox();if(b.x+b.width>r.x+r.width-18)failures.push({text:t.textContent,right:b.x+b.width,bound:r.x+r.width-18});}}return failures;});bounds.push({scene:id,overflow:fit});
 for(const [letter,time] of [['a',12],['b',27]]){await page.evaluate(t=>window.seek(t),time);await page.screenshot({path:path.join(directory,`${id}-${letter}.png`)});}}
 await writeFile(path.join(directory,only?'bounds-correction.json':'bounds.json'),JSON.stringify(bounds,null,2));if(bounds.some(b=>b.overflow.length))throw Error(JSON.stringify(bounds.filter(b=>b.overflow.length)));if(errors.length)throw Error(JSON.stringify(errors));console.log(`${bounds.length*2} chapter stills rendered; SVG panel text bounds passed; no page errors`);
 }else{
 const batchArg=process.argv.find(a=>a.startsWith('--batch='));const batchPath=batchArg?.slice(8)||autoBatchPath,batch=batchPath?JSON.parse(await readFile(batchPath,'utf8')):null;
 const jobs=batch?batch.scenes.filter((job,index)=>index>= (autoStart?batch.scenes.findIndex(s=>s.id===autoStart):0)):[{id:scene,seconds:duration,output:path.resolve(dest),frames:Math.round(duration*30)}];
 for(const job of jobs){const scene=job.id,duration=job.seconds,dest=job.output;
 await page.goto(`http://127.0.0.1:${port}/video/remaster/graphics/scenes.html?scene=${scene}&duration=${duration}`,{waitUntil:'networkidle0'});await page.waitForFunction(()=>window.ready);await page.evaluate(()=>document.fonts.ready.then(()=>true));
 if(still&&!batch){await page.evaluate(t=>window.seek(t),duration*.78);await page.screenshot({path:path.resolve(still)});}
 encoder=spawn('ffmpeg',['-y','-v','error','-f','image2pipe','-vcodec',fullPaint?'png':'mjpeg','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',dest],{stdio:['pipe','inherit','inherit']});const finished=once(encoder,'exit'),started=Date.now();
 for(let frame=0;frame<job.frames;frame++){await page.evaluate(t=>{if(innerWidth!==1920||innerHeight!==1080)throw Error(`Capture viewport changed: ${innerWidth}x${innerHeight}`);window.seek(t)},frame/30);if(fullPaint)await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));const bytes=await page.screenshot(fullPaint?{type:'png',captureBeyondViewport:true,optimizeForSpeed:false}:{type:'jpeg',quality:94,captureBeyondViewport:false,optimizeForSpeed:true});if(!encoder.stdin.write(bytes))await once(encoder.stdin,'drain');if(frame%150===0)console.log(`${scene}: frame ${frame}/${job.frames}, elapsed ${((Date.now()-started)/1000).toFixed(1)}s`);}
 encoder.stdin.end();const [code]=await finished;if(code||errors.length)throw Error(JSON.stringify({code,errors}));console.log(`${scene}: ${duration}s, ${job.frames} frames, no page errors`);
 if(batch){const probe=spawnSync('ffprobe',['-v','error','-show_entries','stream=width,height,r_frame_rate,nb_frames','-show_entries','format=duration','-of','json',dest],{encoding:'utf8'});if(probe.status)throw Error(probe.stderr);job.probe=JSON.parse(probe.stdout);const stream=job.probe.streams[0];if(stream.width!==1920||stream.height!==1080||stream.r_frame_rate!=='30/1'||Number(stream.nb_frames)!==job.frames)throw Error(`${scene} output mismatch`);job.status='verified';await writeFile(batchPath,JSON.stringify(batch,null,2));}
 }
}
}finally{if(encoder&&!encoder.stdin.destroyed)encoder.stdin.end();if(browser)await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
