import {makeScene2D,Circle,Line,Rect,Txt} from '@motion-canvas/2d';import {all,createRef,waitFor} from '@motion-canvas/core';
export default makeScene2D(function*(view){view.fill('#071b1b');const dot=createRef<Circle>(),edge=createRef<Line>();const xs=[-620,0,620],titles=['Fresh challenge','Bound response','Server record'],sub=['One round at a time','Session bearer token','Timing + replay controls'];
view.add(<Txt text="Keep the response in its proper context" x={-90} y={-390} fontFamily="Space" fontSize={69} fill="#f3f0e5"/>);
view.add(<Line ref={edge} points={[[-620,0],[620,0]]} stroke="#1cd6ba" lineWidth={3} end={0}/>);
for(let i=0;i<3;i++)view.add(<Rect x={xs[i]} width={470} height={240} radius={22} fill="#0c2a28" stroke="#38645c" lineWidth={2}><Txt text={titles[i]} y={-45} fontFamily="Space" fontSize={39} fill="#f3f0e5"/><Txt text={sub[i]} y={35} fontFamily="Inter" fontSize={26} fill="#adc7c1"/></Rect>);
view.add(<><Circle ref={dot} x={-620} y={165} size={22} fill="#f3f0e5" opacity={0}/><Txt text="Expected answers stay server-side" y={280} fill="#b89a6a" fontFamily="Inter" fontSize={35}/><Txt text="Relays and assisted solvers remain limitations" y={355} fill="#adc7c1" fontFamily="Inter" fontSize={29}/></>);
yield* edge().end(1,1);yield* dot().opacity(1,.3);yield* dot().position.x(0,1.4);yield* waitFor(.5);yield* dot().position.x(620,1.4);yield* dot().opacity(0,.4);yield* waitFor(3);
});
