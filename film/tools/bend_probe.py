import sys, time, math, numpy as np, cv2
sys.path.insert(0,'.')
import render, glsl, gl3d
r=render.Renderer(); r.spp=1
S=r.scene
cx,cy=300.0,1300.0
outs=[]
def shot(R, cam_pos, target, up=(0,0,1.0)):
    glsl.STATE['bend']=(cx,cy,0.0,float(R)); glsl.STATE['sunp']=(cx,cy,float(R)*0.55,0.8)
    c=np.array(cam_pos,float); f=np.array(target,float)-c; f/=np.linalg.norm(f)
    rr=np.cross(f,up if abs(f[2])<0.98 else (0,1.0,0)); rr/=np.linalg.norm(rr); u=np.cross(rr,f)
    class C: pass
    cam=C(); cam.c=c; cam.R=np.stack([rr,u,f]); cam.t=np.array([cx,cy,0.0]); cam.dist=float(np.linalg.norm(np.array(target)-c))
    S.map_matrices=lambda cm,_S=S: (lambda VP_LVP: VP_LVP)(_mm(cm))
    return cam
def _mm(cam):
    V=gl3d.look(cam.c,cam.R[0],cam.R[1],cam.R[2]); P=gl3d.perspective(5.0,200000.0)
    jx,jy=getattr(S,'jitter',(0,0)); P[0,2]+=jx; P[1,2]+=jy
    f=-gl3d.SUN_DIR; rr=np.cross(f,[0,0,1.0]); rr/=np.linalg.norm(rr); u=np.cross(rr,f)
    LVP=gl3d.ortho(14000,10,40000)@gl3d.look(cam.t+gl3d.SUN_DIR*20000,rr,u,f)
    return P@V,LVP,cam.c
R=4200
for cam_pos,target in (((cx,cy-200,R*1.0),(cx+R*0.9,cy+R*0.3,R*1.6)),((cx-2500,cy-2500,R*0.5),(cx+6000,cy+6000,R*0.9)),((cx,cy-6500,R*0.55),(cx,cy+8000,R*0.4)),((cx,cy-1500,600),(cx,cy+6000,R*0.6))):
    cam=shot(R,cam_pos,target)
    t0=time.time(); img=S.render_map(cam,20.0,render.spring); print(round(time.time()-t0,2))
    outs.append(cv2.resize(img,(960,540)))
cv2.imwrite('build/bend_test.jpg',cv2.cvtColor(np.vstack([np.hstack(outs[:2]),np.hstack(outs[2:])]),cv2.COLOR_RGB2BGR))
