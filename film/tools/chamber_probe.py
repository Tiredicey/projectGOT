import sys, time, numpy as np, cv2, moderngl
sys.path.insert(0,'.')
from timeline import W,H
import chamber, machine as mc
ctx=moderngl.create_standalone_context(backend="egl")
fbo_ms=ctx.framebuffer(color_attachments=[ctx.renderbuffer((W,H),4,samples=4)],depth_attachment=ctx.depth_renderbuffer((W,H),samples=4))
fbo=ctx.framebuffer(color_attachments=[ctx.renderbuffer((W,H),4)])
def read(c):
    ctx.copy_framebuffer(fbo,fbo_ms); return np.frombuffer(fbo.read(components=c,alignment=1),np.uint8).reshape(H,W,c)[::-1]
ch=chamber.Chamber(ctx,fbo_ms,read)
pan=mc.Panel('build/kanal/kanal_01_tabo.mp4')
outs=[]
for lt,cp in ((0.25,0.05),(0.45,0.1),(2.0,0.4),(4.5,0.9)):
    img,_=pan.frame(int(lt*24)+60)
    t0=time.time(); fr=ch.render(5.5+lt,lt,img,1,cp); print('render',round(time.time()-t0,3))
    outs.append(cv2.resize(fr,(960,540)))
cv2.imwrite('build/ch_test.jpg',cv2.cvtColor(np.vstack([np.hstack(outs[:2]),np.hstack(outs[2:])]),cv2.COLOR_RGB2BGR))
