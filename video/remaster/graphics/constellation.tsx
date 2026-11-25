import {makeScene2D,Circle,Line,Txt} from '@motion-canvas/2d';
import {all,createRef,waitFor} from '@motion-canvas/core';
export default makeScene2D(function*(view){
 view.fill('#071b1b');const labels=['Adversarial','Machine-oriented','Self-reference','Social & temporal','Inverse Turing','Anti-thrall','Agency','Counter-coaching','Intent & provenance','Novel reasoning','Governance','LLM-dynamic'];
 const nodes=labels.map(()=>createRef<Circle>()),lines=labels.map(()=>createRef<Line>());
 view.add(<><Txt text="Twelve research lenses" x={-400} y={-400} fill="#f3f0e5" fontFamily="Space" fontSize={74}/></>);
 for(let i=0;i<12;i++){const a=(i/12)*Math.PI*2-Math.PI/2,x=Math.cos(a)*500,y=Math.sin(a)*270;view.add(<><Line ref={lines[i]} points={[[0,0],[x,y]]} stroke="#356c61" lineWidth={2} end={0}/><Circle ref={nodes[i]} x={x} y={y} size={22} fill="#1cd6ba" opacity={0}/><Txt text={labels[i]} x={x+Math.sign(x)*60} y={y} offset={[x<0?1:-1,0]} textAlign={x<0?'right':'left'} fill="#f3f0e5" fontFamily="Inter" fontSize={24} opacity={()=>nodes[i]().opacity()}/></>);}
 view.add(<><Circle size={160} stroke="#b89a6a" lineWidth={2} fill="#0c2a28"/><Txt text="SESSION" fontFamily="Inter" fontSize={27} fill="#f3f0e5"/></>);
 view.add(<Txt text="Different hypotheses. Answers and timing remain session evidence." y={355} fontFamily="Inter" fontSize={30} fill="#adc7c1"/>);
 for(let i=0;i<12;i++)yield* all(nodes[i]().opacity(1,.18),lines[i]().end(1,.18));
 yield* waitFor(2.4);yield* all(...nodes.map(n=>n().size(30,.5)));yield* all(...nodes.map(n=>n().size(22,.5)));yield* waitFor(2.5);
});
