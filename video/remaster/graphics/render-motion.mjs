// One task-owned browser, closed normally. No native game or user browser control.
import {createServer} from 'vite';
import puppeteer from 'puppeteer-core';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import path from 'node:path';
const root=import.meta.dirname;
const scene=process.argv[2]||'constellation', duration=Number(process.argv[3]||8);
const output=path.resolve(process.argv[4]||'video/build/remaster/graphics/constellation.mp4');
const server=await createServer({root,configFile:path.join(root,'vite.config.mjs'),server:{host:'127.0.0.1',port:9237,strictPort:true}});
await server.listen();
let browser,encoder;
try{
 browser=await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--disable-background-networking']});
 const page=await browser.newPage();
 await page.setViewport({width:1920,height:1080,deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
 encoder=spawn('ffmpeg',['-y','-v','warning','-f','image2pipe','-vcodec','png','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','17','-pix_fmt','yuv420p',output],{stdio:['pipe','inherit','inherit']});
 const finished=once(encoder,'exit');
 await page.exposeFunction('emitFrame',async data=>{
   const frame=Buffer.from(data.split(',')[1],'base64');
   if(!encoder.stdin.write(frame))await once(encoder.stdin,'drain');
 });
 await page.goto(`http://127.0.0.1:9237/render.html?scene=${scene}&duration=${duration}`,{waitUntil:'networkidle0'});
 await page.waitForFunction(()=>window.ready===true,{timeout:60000});
 const result=await page.evaluate(()=>window.render());
 encoder.stdin.end();const [code]=await finished;
 if(code!==0||result!==0||errors.length)throw Error(JSON.stringify({code,result,errors}));
 console.log(`Motion Canvas ${scene}: ${duration*30} frames rendered, no browser/encoder errors`);
}finally{
 if(encoder&&!encoder.stdin.destroyed)encoder.stdin.end();
 if(browser)await browser.close();
 await server.close();
}
