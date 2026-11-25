import {makePlugin,ObjectMetaField} from '@motion-canvas/core';
class PipeExporter {
 static id='local-pipe';static displayName='Local frame pipe';
 static meta(){return new ObjectMetaField('options',{})}
 static async create(){return new PipeExporter()}
 async handleFrame(canvas:HTMLCanvasElement){await (window as any).emitFrame(canvas.toDataURL('image/png'));}
 async start(){}async stop(){}
}
export default makePlugin({name:'local-frame-pipe',exporters(){return [PipeExporter]}})();
