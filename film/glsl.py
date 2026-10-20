BEND = """
uniform vec4 bend;
vec3 bendp(vec3 p) {
    if (bend.w <= 0.0) return p;
    vec2 d = p.xy - bend.xy;
    float r = length(d);
    if (r < 1e-3) return p;
    vec2 u = d / r;
    float th = min(r / bend.w, 3.0);
    float rr = bend.w - (p.z - bend.z);
    return vec3(bend.xy + u * rr * sin(th), bend.z + bend.w - rr * cos(th));
}
vec3 bendn(vec3 p, vec3 n) {
    if (bend.w <= 0.0) return n;
    vec2 d = p.xy - bend.xy;
    float r = length(d);
    if (r < 1e-3) return n;
    vec2 u = d / r;
    float th = min(r / bend.w, 3.0);
    vec3 a = vec3(-u.y, u.x, 0.0);
    float c = cos(th), s = sin(th);
    return n * c + cross(a, n) * s + a * dot(a, n) * (1.0 - c);
}
"""

SUNLIGHT = """
uniform vec4 sunp;
vec3 light_dir(vec3 wb, vec3 Ld) {
    if (sunp.w <= 0.0) return Ld;
    return normalize(mix(Ld, normalize(sunp.xyz - wb), sunp.w));
}
"""

IRIS = """
uniform vec4 iris;
void iris_cut(vec3 wf) {
    if (iris.w > 0.5 && length(wf.xy - iris.xy) < iris.z) discard;
}
"""
