// App logic extracted to dedicated file
let socket;
let isDetectionActive=false;
let videoStream=null;
let captureIntervalId=null;
let frameSkipMax=0, frameSkipCount=0;
let imageQuality=75;
let totalFrames=0;let processingVideo=false;

// Elements
const videoFeed=()=>document.getElementById('videoFeed');
const processedVideo=()=>document.getElementById('processedVideo');
const cameraSrc=()=>document.getElementById('cameraSrc');
const placeholder=()=>document.getElementById('placeholder');
const statusEl=()=>document.getElementById('status');
const dotEl=()=>document.getElementById('dot');
const fpsLabel=()=>document.getElementById('fpsLabel');
const objLabel=()=>document.getElementById('objLabel');
const procLabel=()=>document.getElementById('procLabel');
const framesLabel=()=>document.getElementById('framesLabel');
const totalDetLabel=()=>document.getElementById('totalDet');
const progWrap=()=>document.getElementById('uploadWrap');
const progBar=()=>document.getElementById('progressBar');
const progText=()=>document.getElementById('progressText');
const progStats=()=>document.getElementById('progressStats');
const detectStats=()=>document.getElementById('detectionStats');

function setStatus(kind,text){
  statusEl().textContent=text;
  dotEl().className='dot '+(kind||'ok');
}

function show(el){el.style.display='block'}
function hide(el){el.style.display='none'}

function initSocket(){
  socket=io();
  socket.on('connect',()=>setStatus('ok','Connected'));
  socket.on('disconnect',()=>{setStatus('warn','Disconnected'); stopCapture();});
  socket.on('reconnect',()=>setStatus('ok','Reconnected'));

  socket.on('processed_frame',(data)=>{
    if(!isDetectionActive||!data.image) return;
    videoFeed().src='data:image/jpeg;base64,'+data.image;
    show(videoFeed()); hide(processedVideo()); hide(placeholder());
    fpsLabel().textContent=(data.fps||0);
    objLabel().textContent=(data.detections||0);
    procLabel().textContent=((data.processing_time||0)+' ms');
    totalFrames++; framesLabel().textContent=totalFrames;
  });

  socket.on('video_frame',(data)=>{
    if(!processingVideo||!data.image) return;
    videoFeed().src='data:image/jpeg;base64,'+data.image;
    show(videoFeed()); hide(processedVideo()); hide(placeholder());
    const p=data.progress||0; progBar().style.width=p+'%';
    progText().textContent=`Frame ${data.frame_number}/${data.total_frames}`;
    progStats().textContent=`${data.fps||0} FPS • ${data.detections||0} det • ${data.elapsed_time||0}s`;
    detectStats().textContent=`Total detections: ${data.total_detections||0}`;
    totalDetLabel().textContent=(data.total_detections||0);
  });

  socket.on('video_complete',(data)=>{
    processingVideo=false; hide(progWrap());
    if(data.output_url){ hide(videoFeed()); show(processedVideo()); processedVideo().src=data.output_url; processedVideo().play().catch(()=>{}); }
  });

  socket.on('processing_stopped',()=>{processingVideo=false; hide(progWrap());});
  socket.on('error',(d)=>{console.error('socket error',d); setStatus('err','Error');});
}

async function startCamera(){
  try{
    // Prefer user-facing cam; allow fallback
    const constraints={video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:30},facingMode:'user'},audio:false};
    videoStream=await navigator.mediaDevices.getUserMedia(constraints);
    const v=cameraSrc(); v.srcObject=videoStream; v.muted=true; v.playsInline=true;
    await v.play();
    await waitForReady(v);
    isDetectionActive=true; totalFrames=0; framesLabel().textContent='0';
    startCapture(v);
    setStatus('ok','Processing');
  }catch(e){
    console.error('getUserMedia failed',e);
    setStatus('err','Camera blocked');
    alert('Cannot access camera. Allow permission in browser settings.');
  }
}

function stopCamera(){
  isDetectionActive=false; stopCapture();
  if(videoStream){ videoStream.getTracks().forEach(t=>t.stop()); videoStream=null; }
}

function waitForReady(video){
  return new Promise(res=>{
    if(video.readyState>=2) return res();
    video.onloadedmetadata=()=>res();
    video.onplaying=()=>res();
  });
}

function startCapture(video){
  const canvas=document.createElement('canvas');
  const ctx=canvas.getContext('2d');
  function tick(){
    if(!isDetectionActive) return;
    // frame skip
    if(frameSkipCount<frameSkipMax){ frameSkipCount++; return; }
    frameSkipCount=0;
    try{
      canvas.width=video.videoWidth||640; canvas.height=video.videoHeight||480;
      ctx.drawImage(video,0,0,canvas.width,canvas.height);
      const data=canvas.toDataURL('image/jpeg',Math.max(0.4,Math.min(0.95,imageQuality/100)));
      socket.emit('webcam_frame',{image:data});
    }catch(e){ console.warn('capture error',e); }
  }
  // ~12fps stable
  captureIntervalId=setInterval(tick,83);
}

function stopCapture(){ if(captureIntervalId){ clearInterval(captureIntervalId); captureIntervalId=null; } }

async function handleUpload(file){
  const form=new FormData(); form.append('file',file);
  show(progWrap()); progBar().style.width='0%'; progText().textContent='Uploading...'; progStats().textContent=''; detectStats().textContent='';
  const res=await fetch('/upload',{method:'POST',body:form});
  const json=await res.json(); if(!res.ok){ alert(json.error||'Upload failed'); hide(progWrap()); return; }
  if(json.type==='image'){
    hide(processedVideo()); show(videoFeed()); hide(placeholder());
    videoFeed().src='data:image/jpeg;base64,'+json.result; hide(progWrap()); objLabel().textContent=json.detections||0;
  }else if(json.type==='video'){
    processingVideo=true; show(videoFeed()); hide(processedVideo()); hide(placeholder());
    progText().textContent='Starting video processing...'; progStats().textContent='Initializing...';
    socket.emit('process_video',{filepath:json.filepath});
  }
}

function bindUI(){
  document.getElementById('btnStart').onclick=()=>startCamera();
  document.getElementById('btnStop').onclick=()=>{stopCamera(); setStatus('ok','Connected');};
  document.getElementById('btnStopProc').onclick=()=>socket.emit('stop_processing');

  const upZone=document.getElementById('uploadZone');
  const input=document.getElementById('fileInput');
  upZone.onclick=()=>input.click();
  upZone.ondragover=(e)=>{e.preventDefault(); upZone.classList.add('dragover');};
  upZone.ondragleave=()=>upZone.classList.remove('dragover');
  upZone.ondrop=(e)=>{e.preventDefault(); upZone.classList.remove('dragover'); if(e.dataTransfer.files?.length){handleUpload(e.dataTransfer.files[0])}};
  input.onchange=(e)=>{ if(e.target.files?.length){handleUpload(e.target.files[0])} };

  const conf=document.getElementById('confidence');
  conf.oninput=(e)=>{ document.getElementById('confidenceVal').textContent=e.target.value; };
  const skip=document.getElementById('frameSkip');
  skip.oninput=(e)=>{ frameSkipMax=parseInt(e.target.value||'0',10); document.getElementById('frameSkipVal').textContent=skip.value; };
  const qual=document.getElementById('quality');
  qual.oninput=(e)=>{ imageQuality=parseInt(e.target.value||'75',10); document.getElementById('qualityVal').textContent=qual.value; };
}

window.addEventListener('DOMContentLoaded',()=>{ initSocket(); bindUI(); setStatus('warn','Connecting...'); });
