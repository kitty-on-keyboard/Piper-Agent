// The activity orb: a raymarched glass bead that shows what the run is doing.
//
// Ported from LM_Pipe v1's "Cymatic Loop" (media/sidebar.js), which was a spring-driven
// WebGL2 shader orb. The physics, the OKLCH state palette and the transition choreography
// are the parts worth keeping and are carried over largely intact. The RENDERING is not:
// v1's orb was an emissive blob with a Fresnel rim, and it read as a glow rather than as an
// object. This one is glass.
//
// WHAT CHANGED, AND WHY EACH CHANGE IS VISIBLE
//
// 1. It actually refracts. v1 declared `u_ior` and never used it -- there was no
//    transmission term at all. Here the ray refracts in, marches the interior chord, and
//    refracts out through three slightly different indices, so the exit direction splits
//    into red/green/blue. That colour fringing at the rim is the single strongest "this is
//    glass" cue, and no amount of glow substitutes for it.
//
// 2. Beer-Lambert absorption over the measured chord. The bead is tinted by how far the
//    light travelled through it, so the thin edges are pale and the middle is saturated.
//    That is what makes it read as a VOLUME rather than a shell.
//
// 3. The silhouette is analytically antialiased from the closest-approach distance of the
//    sphere trace, giving a true ~1px edge. v1 faded the alpha out with a Fresnel ramp,
//    which is why it looked soft and slightly out of focus at every size.
//
// 4. Bounding-sphere entry/exit. The march runs only across the interval that can contain
//    the surface instead of from the camera to t=3.2. Most pixels now cost nothing, which
//    is what pays for 1-3, and the steps that do run are spent where the detail is.
//
// 5. Motion phases are ACCUMULATED on the CPU (`u_flow`, `u_breath`) rather than computed
//    as `u_time * rate` in the shader. v1's rates changed with the agent state, so every
//    state change instantaneously rescaled the whole phase and the noise field visibly
//    jumped. Integrating the rate makes the field continuous across a state change: the
//    orb speeds up, it does not teleport.
//
// 6. Two-light studio (tight key, broad accent fill) through a GGX lobe instead of a
//    pow(dot) blob, plus thin-film iridescence on the Fresnel rim.
//
// The whole thing degrades in two steps: no WebGL2, or a reduced-motion preference, both
// fall back to a CSS bead that is tinted by the same state palette.

/** Rendered size of the bead, in CSS pixels. Big enough that the key highlight and the
 *  dispersion fringe are more than one pixel each -- below about 24px the glass reads as a
 *  dot and everything above is wasted work. */
const ORB_SIZE = 34;

/** The fragment shader. Composited premultiplied over the editor background, so it never
 *  draws its own backdrop: alpha is coverage, and the colour is unpremultiplied before the
 *  sRGB encode so the halo keeps full chroma instead of being dimmed twice. */
const FRAGMENT_SHADER = `#version 300 es
precision highp float;
out vec4 fragColor;

uniform vec2  u_res;
uniform float u_time;       // raw seconds -- dither, ripple, hue drift
uniform float u_flow;       // INTEGRATED flow phase (see note 5 above)
uniform float u_breath;     // INTEGRATED breath phase
uniform float u_intensity;
uniform float u_pulse;
uniform float u_spec;
uniform float u_energy;
uniform float u_scale;
uniform float u_ior;
uniform vec3  u_color;      // state colour, linear sRGB
uniform vec3  u_accent;     // rim / fill colour, linear sRGB
uniform float u_bg;         // background luminance, 0..1
uniform float u_ripple;
uniform float u_impulse;

// Pre-normalized so they can be const: normalize() is not a constant expression here.
const vec3 KEY  = vec3(-0.4203, 0.6804, 0.6004);   // tight catch-light, upper left
const vec3 FILL = vec3( 0.5498, 0.2999, 0.7797);   // broad accent fill, upper right
const float PI  = 3.14159265;

// -- noise ------------------------------------------------------------------

vec3 hash33(vec3 p) {
    p  = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx) * 2.0 - 1.0;
}

float hash21(vec2 p) {
    p  = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// Gradient (Perlin-style) noise, roughly [-0.5, 0.5].
float gNoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(
        mix(mix(dot(hash33(i),             f            ),
                dot(hash33(i+vec3(1,0,0)), f-vec3(1,0,0)), u.x),
            mix(dot(hash33(i+vec3(0,1,0)), f-vec3(0,1,0)),
                dot(hash33(i+vec3(1,1,0)), f-vec3(1,1,0)), u.x), u.y),
        mix(mix(dot(hash33(i+vec3(0,0,1)), f-vec3(0,0,1)),
                dot(hash33(i+vec3(1,0,1)), f-vec3(1,0,1)), u.x),
            mix(dot(hash33(i+vec3(0,1,1)), f-vec3(0,1,1)),
                dot(hash33(i+vec3(1,1,1)), f-vec3(1,1,1)), u.x), u.y), u.z);
}

// 3 octaves, lattice rotated between them so the axes of the grid never show.
float fbm(vec3 p) {
    mat3 m = mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);
    float v = 0.0, a = 0.5;
    v += a * gNoise(p); p = m * p * 2.02; a *= 0.5;
    v += a * gNoise(p); p = m * p * 2.02; a *= 0.5;
    v += a * gNoise(p);
    return v;
}

// -- geometry ---------------------------------------------------------------

float energy()    { return clamp(u_energy, 0.0, 1.2); }
float radius()    { return 0.30 * u_scale; }
float warpAmp()   { return mix(0.055, 0.185, energy()); }
float breathAmp() { return mix(0.014, 0.048, energy()) * clamp(u_pulse * 8.0 + 0.6, 0.6, 1.5); }

// A sphere is guaranteed to contain the warped surface once it clears the largest
// displacement any term can apply. Conservative on purpose: the march is clipped to this
// interval, so underestimating it would clip the bead itself.
float boundR() {
    return radius() + warpAmp() * 0.92 + breathAmp() * 1.45 + u_ripple * 0.03 + 0.006;
}

// Domain warp. THREE independent noise fields, each advancing on a different phase
// multiple -- which is what produces apparent rotation with no rotation matrix anywhere,
// and therefore no visible gyroscope axis.
vec3 warp(vec3 p) {
    float sc = 1.75;
    return vec3(
        fbm(p * sc + vec3(u_flow,        u_flow * 0.71, 0.00)),
        fbm(p * sc + vec3(u_flow * 0.83, 4.73,          u_flow * 0.91)),
        fbm(p * sc + vec3(0.00,          u_flow * 0.77, u_flow * 1.15))
    ) * warpAmp();
}

float sdf(vec3 p) {
    float d  = length(p + warp(p)) - radius();
    float ba = breathAmp();
    // Two breaths at an irrational-ish ratio, so the cycle never visibly repeats.
    d -= ba * sin(u_breath);
    d -= ba * 0.37 * sin(u_breath * 1.8177 + 1.31);
    if (u_ripple > 0.01) {
        float r = length(p);
        d -= u_ripple * 0.024 * sin(r * 26.0 - u_time * 9.5) * exp(-r * 5.5);
    }
    return d;
}

// Tetrahedral normal: 4 evaluations, no branches, correct on a warped field.
vec3 normalAt(vec3 p, float h) {
    const vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * sdf(p + k.xyy * h) +
        k.yyx * sdf(p + k.yyx * h) +
        k.yxy * sdf(p + k.yxy * h) +
        k.xxx * sdf(p + k.xxx * h)
    );
}

// -- shading ----------------------------------------------------------------

// What the glass has to look at. There is no scene, so this IS the scene: a soft vertical
// studio gradient, one small bright key, and an accent-coloured fill. Sampling it through
// three dispersed exit directions is what puts colour in the rim.
// What the glass has to look at, and the reason this file has an environment at all.
//
// The orb composites over a flat editor background, so refracting the BACKDROP buys
// nothing -- bending a uniform colour gives back the same uniform colour. Everything that
// makes glass look like glass has to be in here instead. A dark sky with one hot key gave
// a bead with nothing inside it, which is why the first version could only ever read as a
// lamp: with no structure behind it, every pixel of the body was emission.
//
// So this is a small studio. The bright horizontal SOFTBOX is the load-bearing part: a lens
// bends a straight bright edge into a curved streak, and that streak is what the eye reads
// as "I am looking through something" -- faster and more reliably than any specular.
vec3 env(vec3 d) {
    float up = d.y * 0.5 + 0.5;

    vec3 floorC = vec3(0.010, 0.012, 0.018);
    vec3 ceilC  = u_accent * 0.30 + vec3(0.020, 0.024, 0.032);
    vec3 sky    = mix(floorC, ceilC, smoothstep(0.30, 0.92, up));

    sky += vec3(1.00, 0.98, 0.96) * exp(-pow((up - 0.62) / 0.105, 2.0)) * 2.4;

    // A second, dimmer, tilted source. Its only job is to break the symmetry of the first:
    // a perfectly symmetric streak looks like a printed highlight, not a reflection.
    float az = atan(d.x, d.z);
    sky += u_accent * exp(-pow((up - 0.33) / 0.10, 2.0))
                    * (0.55 + 0.45 * cos(az + 0.9)) * 0.55;

    // Small and hot rather than broad and bright: a tight source is what survives being
    // bent through the glass as a distinct glint instead of a smear.
    sky += vec3(1.00, 0.985, 0.955) * pow(max(dot(d, KEY),  0.0), 380.0) * 15.0;
    sky += u_accent                 * pow(max(dot(d, FILL), 0.0),  22.0) * 0.85;
    return sky;
}

float ggx(vec3 n, vec3 v, vec3 l, float rough) {
    vec3  h   = normalize(v + l);
    float a   = max(rough * rough, 1e-3);
    float ndh = max(dot(n, h), 0.0);
    float den = ndh * ndh * (a * a - 1.0) + 1.0;
    return (a * a) / (PI * den * den);
}

// Thin-film interference, cheaply: a cosine palette driven by the Fresnel term.
vec3 irid(float t) {
    return 0.5 + 0.5 * cos(6.28318 * (t * 1.25 + vec3(0.0, 0.33, 0.67)) + u_time * 0.12);
}

// Luminance-preserving Reinhard. Keeps the state hue at full purity instead of letting the
// highlights drag it toward white, which is what an ACES-style curve does here.
vec3 tonemapLuma(vec3 c, float W) {
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    if (luma < 1e-5) return c;
    float nLuma = luma * (1.0 + luma / (W * W)) / (1.0 + luma);
    return c * (nLuma / luma);
}

vec3 linearToSrgb(vec3 c) {
    return mix(12.92 * c,
               1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055,
               step(vec3(0.0031308), c));
}

// -- main -------------------------------------------------------------------

void main() {
    vec2  frag = gl_FragCoord.xy;
    vec2  uv   = (frag - 0.5 * u_res) / u_res.y;
    float px   = 1.0 / u_res.y;             // pixel footprint, in uv units
    float E    = energy();

    vec3 ro = vec3(0.0, 0.0, 1.6);
    vec3 rd = normalize(vec3(uv, -1.0));

    float drift = 0.018 * sin(u_time * 0.45) + 0.012 * sin(u_time * 1.33 + 0.8);
    vec3  base  = max(u_color * (1.0 + drift), vec3(0.001));
    float bgA   = mix(1.06, 0.74, clamp(u_bg, 0.0, 1.0));

    // -- halo, analytically ------------------------------------------------
    // Distance from the ray to the mean sphere at closest approach. v1 spent a 24-step
    // march finding this; it is a dot product.
    float bb   = dot(ro, rd);
    float tc   = max(-bb, 0.0);
    float gd   = max(length(ro + rd * tc) - radius(), 0.0);
    float sig  = mix(0.027, 0.020, clamp(u_bg, 0.0, 1.0));
    float glow = exp(-gd / sig) * clamp(u_intensity, 0.0, 1.4) * bgA;

    vec3  col = base * (0.55 + 0.75 * glow) * glow;
    float alf = glow * (0.20 + 0.20 * u_impulse);

    // -- surface -----------------------------------------------------------
    float bR = boundR();
    float cc = dot(ro, ro) - bR * bR;
    float hh = bb * bb - cc;

    if (hh > 0.0) {
        float sq   = sqrt(hh);
        float t    = max(-bb - sq, 0.006);
        float tEnd = -bb + sq;
        float minD = 1e9;
        bool  hit  = false;
        vec3  hp   = vec3(0.0);

        for (int i = 0; i < 48; i++) {
            vec3  p = ro + rd * t;
            float d = sdf(p);
            minD = min(minD, d);
            if (d < 0.0006) { hit = true; hp = p; break; }
            t += max(d * 0.90, 0.0016);
            if (t > tEnd) break;
        }

        // Analytic coverage: the closest approach IS the perpendicular distance to the
        // silhouette, so one smoothstep over a pixel width gives a true antialiased edge.
        float sd  = hit ? -px : minD;
        float cov = 1.0 - smoothstep(0.0, 1.15 * px, sd);

        if (hit) {
            vec3 n0 = normalAt(hp, 0.0022);

            // Surface tension: high-frequency detail perturbing the normal only. Adding it
            // to the SDF instead would force a much smaller march step for the same look.
            //
            // Applied to the REFLECTIVE normal only. Refracting through it scattered the
            // interior caustic into blotches -- a bent ray amplifies a normal wobble over
            // the whole chord, so the one bright focus the key should make came out as
            // three or four smeared patches.
            vec3 mn = vec3(
                gNoise(hp * 13.0 + vec3(u_flow * 0.6, 0.00,          0.00)),
                gNoise(hp * 13.0 + vec3(0.00,         u_flow * 0.6, 11.30)),
                gNoise(hp * 13.0 + vec3(7.10,         0.00,          u_flow * 0.6))
            );
            vec3 n = normalize(n0 + mn * (0.035 + 0.045 * E));

            float ndv = clamp(dot(n, -rd), 0.0, 1.0);
            float F   = 0.045 + 0.955 * pow(1.0 - ndv, 5.0);

            // Refract in through the smooth normal, and measure the chord.
            vec3  ri  = refract(rd, n0, 1.0 / u_ior);
            float tin = 0.006;
            for (int i = 0; i < 14; i++) {
                float d = sdf(hp + ri * tin);
                if (d > -0.0008) break;
                tin += max(-d * 0.94, 0.006);
            }
            float thick = clamp(tin, 0.008, 1.2);

            // Exit normal from the un-warped sphere gradient. Exact enough for a direction
            // that is about to be scattered anyway, and 3 noise evaluations instead of 12.
            vec3 xp = hp + ri * thick;
            vec3 nx = normalize(xp + warp(xp));

            // Dispersion: blue bends hardest. Three exit directions, one channel each.
            // Spread well past a real crown glass -- physical Abbe numbers put the fringe
            // below a pixel at 34px, and an effect nobody can see is not worth the cost.
            vec3 oR = refract(ri, -nx, u_ior * 0.965);
            vec3 oG = refract(ri, -nx, u_ior);
            vec3 oB = refract(ri, -nx, u_ior * 1.035);
            // Total internal reflection returns a zero vector; fall back to a mirror.
            if (dot(oR, oR) < 0.5) oR = reflect(ri, -nx);
            if (dot(oG, oG) < 0.5) oG = reflect(ri, -nx);
            if (dot(oB, oB) < 0.5) oB = reflect(ri, -nx);

            vec3 trans = vec3(env(oR).r, env(oG).g, env(oB).b);

            // Beer-Lambert: the glass absorbs everything its own colour does not carry, in
            // proportion to how far the light travelled through it.
            vec3 baseN  = base / max(max(base.r, max(base.g, base.b)), 1e-4);
            vec3 absorb = (vec3(1.0) - baseN) * mix(6.5, 3.6, E);
            trans *= exp(-absorb * thick * 3.5);

            // Interior caustics: two samples of the flow field along the chord, so the
            // filaments drift with the same motion as the surface. Narrow thresholds -- a
            // wide one gives an even haze, and a caustic that is everywhere is not a caustic.
            float cs = smoothstep(0.10, 0.26, fbm((hp + ri * thick * 0.45) * 3.2 +
                                                  vec3(u_flow * 0.8, 0.0, u_flow * 0.4)))
                     + smoothstep(0.10, 0.26, fbm((hp + ri * thick * 0.80) * 4.4 +
                                                  vec3(0.0, u_flow * 0.9, 0.0)));
            float caustic = cs * 0.5 * smoothstep(0.02, 0.26, thick);

            // Its own light. Deliberately a MINORITY of the final colour: the bead is a lens
            // first and a lamp second, and the earlier balance -- a broad white core over a
            // bright body -- is exactly what made it read as a glowing ball instead of glass.
            vec3 emit = base * pow(ndv, mix(0.50, 0.28, E)) * mix(0.14, 0.30, E) * u_intensity;
            // Hot core kept small and only lightly desaturated, so it is a highlight in the
            // depth of the bead rather than a white wash across it.
            emit += mix(base * 1.6, vec3(1.00, 0.97, 0.93), 0.30)
                    * pow(ndv, 9.0) * u_intensity * 0.30;
            emit += base * caustic * (0.30 + 0.40 * E) * u_intensity;

            // THE glass cue, and the cheapest one: the environment mirrored off the front
            // face, weighted by Fresnel. It is nearly nothing head-on and takes over at the
            // edge, which is what draws the hard bright ring a lit bead has and a lit ball
            // does not. The GGX lobes stay, at low weight, for the broadened sheen around it.
            float rough = mix(0.030, 0.075, E);
            vec3  spec  = env(reflect(rd, n)) * F * 1.15
                        + vec3(1.00, 0.99, 0.975) * ggx(n, -rd, KEY,  rough)       * u_spec * 0.55
                        + u_accent                * ggx(n, -rd, FILL, rough * 4.0) * u_spec * 0.22;
            spec = min(spec, vec3(40.0));

            vec3 rim = mix(u_accent, irid(F + 0.15 * E), 0.55)
                       * F * mix(1.00, 1.55, clamp(u_bg, 0.0, 1.0)) * u_intensity * 1.25;

            // Just inside the silhouette, everything the glass transmits has been bent away
            // from the eye, so a real bead is DARKEST in a band there -- and then abruptly
            // brightest right at the edge. Without this trough the body runs straight into
            // the rim and the whole thing reads as one lit surface.
            float band = smoothstep(0.26, 0.60, F) * (1.0 - smoothstep(0.70, 0.93, F));

            vec3 body = (trans * (1.0 - F) * mix(0.85, 1.25, E) + emit) * (1.0 - 0.55 * band);

            vec3 surf = body + spec * bgA + rim
                      + base * u_impulse * 0.55 * pow(ndv, 1.2);

            // Alpha is what makes it glass rather than a sticker: the body is only partly
            // opaque so the editor shows through it, while the Fresnel rim and the caustics
            // -- the parts that are actually reflecting light at you -- go solid.
            float lum = dot(surf, vec3(0.2126, 0.7152, 0.0722));
            float sa  = cov * clamp(0.56 + 0.42 * F + 0.30 * caustic + 0.45 * lum, 0.0, 1.0);

            col = col * 0.30 + surf;
            alf = max(alf, sa);
        }
    }

    // Unpremultiply before the encode: alpha carries the falloff, so leaving it in the
    // colour too would dim the halo twice and desaturate it.
    float a = clamp(alf, 0.0, 1.0) * clamp(u_intensity * 4.0, 0.0, 1.0);
    vec3  c = col / max(a, 1e-4);
    // White point. v1's was under 1.0, which meant anything moderately lit clipped straight
    // to white and took the state hue with it. Held well above the body's own radiance so
    // only the key glint reaches white.
    c = tonemapLuma(c, mix(1.90, 2.90, E) * mix(1.0, 0.82, clamp(u_bg, 0.0, 1.0)));
    vec3 srgb = linearToSrgb(c) + (hash21(frag + fract(u_time * 17.13)) - 0.5) / 255.0;
    fragColor = vec4(srgb * a, a);
}`;

/** The orb's stylesheet. Sized from one variable so the markup and the canvas backing
 *  store cannot disagree about how big it is. */
export function orbStyles(): string {
  return `
:root { --orb-size: ${ORB_SIZE}px; }

#headRow { display: flex; align-items: center; gap: 11px; padding-right: 26px; }
#headText { flex: 1; min-width: 0; }

#orb {
  width: var(--orb-size); height: var(--orb-size);
  flex: none; display: block; background: transparent;
  pointer-events: none;
  opacity: .74;
  transition: opacity .45s var(--ease);
  filter: drop-shadow(0 1px 4px rgba(0,0,0,.22));
}
body.busy #orb { opacity: 1; }

/* No WebGL2, or a reduced-motion preference. Same palette, one element, no loop -- the
   state is still legible, which is the part that matters. */
#orb.fb {
  border-radius: 50%;
  background:
    radial-gradient(circle at 37% 31%,
      color-mix(in srgb, var(--orb-tint, var(--accent)) 25%, #fff) 0%,
      var(--orb-tint, var(--accent)) 44%,
      color-mix(in srgb, var(--orb-tint, var(--accent)) 45%, transparent) 74%,
      transparent 80%);
  box-shadow: 0 0 11px 1px color-mix(in srgb, var(--orb-tint, var(--accent)) 34%, transparent);
  transition: background .45s var(--ease), box-shadow .45s var(--ease), opacity .45s var(--ease);
}
body.busy #orb.fb { animation: orbBreathe 2.6s ease-in-out infinite; }
@keyframes orbBreathe { 0%,100% { transform: scale(.94); } 50% { transform: scale(1.03); } }

@media (prefers-reduced-motion: reduce) {
  #orb.fb { animation: none !important; }
}
`;
}

/** The canvas. Always a canvas: the fallback paints the same element with CSS rather than
 *  swapping the node, so nothing downstream has to care which path is live. */
export function orbMarkup(): string {
  return `<canvas id="orb" aria-hidden="true"></canvas>`;
}

/** The orb's view script, as source text for the one nonced <script> the CSP allows.
 *
 *  Exposes `window.__orb` = { state(name), impulse(kind), read() }. `read()` is there for
 *  the preview harness in scripts/preview-orb.js, which drives every state in turn so the
 *  thing can be looked at outside the editor. */
export function orbScript(): string {
  return `
(function () {
  // -- spring ---------------------------------------------------------------
  // Substepped at 1ms so a dropped frame cannot make it explode: a 32ms dt integrated in
  // one go with these stiffnesses overshoots badly.
  function Spring(v, k, d) { this.value = v; this.velocity = 0; this.target = v; this.k = k; this.d = d; }
  Spring.prototype.update = function (dt) {
    var left = Math.min(dt, 0.032), step = 0.001;
    while (left > 0) {
      var h = Math.min(left, step);
      this.velocity += (-this.k * (this.value - this.target) - this.d * this.velocity) * h;
      this.value += this.velocity * h;
      left -= h;
    }
    return this.value;
  };
  Spring.prototype.settled = function (eps) {
    eps = eps || 0.0008;
    return Math.abs(this.value - this.target) < eps && Math.abs(this.velocity) < eps;
  };

  // -- OKLCH ----------------------------------------------------------------
  // The palette is authored in OKLCH and interpolated there, so a hue change sweeps around
  // the hue circle at even lightness instead of dipping through grey the way an sRGB lerp
  // between two saturated colours does.
  function oklabToLinear(L, a, b) {
    var l = L + 0.3963377774 * a + 0.2158037573 * b;
    var m = L - 0.1055613458 * a - 0.0638541728 * b;
    var s = L - 0.0894841775 * a - 1.2914855480 * b;
    l = l * l * l; m = m * m * m; s = s * s * s;
    return [
      4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    ];
  }
  function oklchToLinear(c) {
    var h = c.H * Math.PI / 180;
    return oklabToLinear(c.L, c.C * Math.cos(h), c.C * Math.sin(h));
  }
  function linearToSrgb8(x) {
    var v = x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(Math.max(x, 0), 1 / 2.4) - 0.055;
    return Math.round(Math.min(1, Math.max(0, v)) * 255);
  }
  function oklchToCss(c) {
    var l = oklchToLinear(c);
    return 'rgb(' + linearToSrgb8(l[0]) + ',' + linearToSrgb8(l[1]) + ',' + linearToSrgb8(l[2]) + ')';
  }
  function lerpHue(a, b, t) { return (a + (((b - a + 540) % 360) - 180) * t + 360) % 360; }
  function lerpOklch(a, b, t) {
    return { L: a.L + (b.L - a.L) * t, C: a.C + (b.C - a.C) * t, H: lerpHue(a.H, b.H, t) };
  }
  // The rim/fill colour: the same hue rotated a little and lifted, so the dispersion at the
  // edge is a relative of the body colour rather than an unrelated second colour.
  function accentOf(c) { return { L: Math.min(0.95, c.L + 0.12), C: c.C * 0.88, H: c.H + 38 }; }

  // -- the palette ----------------------------------------------------------
  // One entry per thing the run can be doing. Hue is the whole message: nothing else about
  // the orb tells you WHICH activity it is.
  var STATES = {
    IDLE:     { L: 0.70, C: 0.10, H: 250, intensity: 0.58, flow: 0.10, breath: 0.55, pulse: 0.045, spec: 0.75, energy: 0.12 },
    THINKING: { L: 0.74, C: 0.26, H: 285, intensity: 0.95, flow: 0.42, breath: 0.95, pulse: 0.150, spec: 0.88, energy: 0.55 },
    WRITING:  { L: 0.82, C: 0.17, H: 215, intensity: 1.00, flow: 0.30, breath: 1.10, pulse: 0.120, spec: 0.95, energy: 0.40 },
    TOOL:     { L: 0.80, C: 0.27, H:  55, intensity: 1.12, flow: 0.78, breath: 1.35, pulse: 0.260, spec: 0.98, energy: 1.00 },
    WAITING:  { L: 0.78, C: 0.24, H:  20, intensity: 0.92, flow: 0.16, breath: 1.70, pulse: 0.200, spec: 0.90, energy: 0.30 },
    DONE:     { L: 0.85, C: 0.21, H: 155, intensity: 0.80, flow: 0.09, breath: 0.50, pulse: 0.035, spec: 0.95, energy: 0.16 },
    FAILED:   { L: 0.68, C: 0.25, H:  27, intensity: 0.78, flow: 0.12, breath: 0.60, pulse: 0.050, spec: 0.85, energy: 0.22 }
  };
  var EXHALE = { L: 0.92, C: 0.23, H: 155 };

  // How each transition is PLAYED, not just what it lands on. A run that ends should exhale
  // and settle; one that starts should wake; a tool call should hit.
  var CHOREO = {
    'IDLE>THINKING':    { ms: 520, wake: 1 },
    'THINKING>TOOL':    { ms: 380, ripple: 0.9 },
    'WRITING>TOOL':     { ms: 380, ripple: 0.9 },
    'THINKING>WRITING': { ms: 420 },
    '*>DONE':           { ms: 640, collect: 1, exhale: 560 },
    '*>FAILED':         { ms: 640, collect: 1 },
    '*>WAITING':        { ms: 500, wake: 0.6 },
    '*>IDLE':           { ms: 900, relax: 1 }
  };
  function choreo(from, to) { return CHOREO[from + '>' + to] || CHOREO['*>' + to] || { ms: 560 }; }

  // Some states are OVER before they have been seen: a tool call can return in 40ms, a
  // verification result is followed immediately by the next token. Each of these holds the
  // orb for long enough to register, and whatever wanted the orb next is queued rather than
  // dropped. Without it the amber and the red simply never appear -- v1 shipped that bug
  // for TOOL and only ever fixed that one case.
  var HOLD = { TOOL: 420, FAILED: 900, DONE: 640 };

  // -- runtime state --------------------------------------------------------
  var stateId = 'IDLE', pending = null, holdUntil = 0, holdTimer = null;
  var from = STATES.IDLE, to = STATES.IDLE, mix = new Spring(1, 70, 15);
  var springs = {
    intensity: new Spring(STATES.IDLE.intensity, 100, 17),
    flow:      new Spring(STATES.IDLE.flow,       90, 16),
    breath:    new Spring(STATES.IDLE.breath,     90, 16),
    pulse:     new Spring(STATES.IDLE.pulse,      90, 16),
    spec:      new Spring(STATES.IDLE.spec,       90, 16),
    energy:    new Spring(STATES.IDLE.energy,     90, 16),
    scale:     new Spring(1.0,                    80, 15)
  };
  var impToken = 0, impTool = 0, impRipple = 0, lastToken = 0;
  var transitionUntil = 0, bgLuma = 0.12, wake = null, canvas = null;
  // Motion phases live out here rather than inside the renderer so the still-capture hook
  // at the bottom of this file can pin them.
  var animTime = 0, flowPhase = 0, breathPhase = 0, frozen = false;

  // Stiffness that settles in about the requested time at zeta ~0.85.
  function stiffnessFor(ms) { var w = 2 * Math.PI / Math.max(0.2, ms / 1000); return w * w * 0.35; }

  function applyState(name, ch) {
    var p = STATES[name] || STATES.IDLE;
    var k = stiffnessFor(ch.ms), d = Math.sqrt(k) * 1.7;
    for (var key in springs) { springs[key].k = k; springs[key].d = d; }
    mix.k = k; mix.d = d;

    springs.intensity.target = p.intensity;
    springs.flow.target      = p.flow;
    springs.breath.target    = p.breath;
    springs.pulse.target     = p.pulse;
    springs.spec.target      = p.spec;
    springs.energy.target    = p.energy;

    if (ch.wake) {
      springs.intensity.target = Math.min(1.20, springs.intensity.target + 0.12 * ch.wake);
      springs.pulse.target     = Math.min(0.35, springs.pulse.target + 0.06 * ch.wake);
    }
    if (ch.ripple) impRipple = Math.max(impRipple, 0.9 * ch.ripple);
    if (ch.collect) {
      springs.energy.target = Math.max(0.14, springs.energy.target * 0.55);
      springs.scale.target = 0.96;
    }
    if (ch.exhale) {
      springs.intensity.target = Math.min(1.25, STATES.DONE.intensity + 0.35);
      springs.spec.target = 1.05;
      springs.scale.target = 1.05;
    }
    if (ch.relax) springs.scale.target = 1.0;
  }

  function beginColour(next) {
    from = lerpOklch(from, to, mix.value);
    to = next;
    mix.value = 0; mix.target = 1;
  }

  function setState(name) {
    if (!STATES[name]) name = 'IDLE';
    if (name === stateId) return;

    var now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    // WAITING is the one thing a hold never blocks: it means the run has stopped and is
    // asking the user something, which must never be delayed behind a decorative flash.
    if (now < holdUntil && name !== 'WAITING') {
      pending = name;
      return;
    }

    var prev = stateId;
    stateId = name;
    pending = null;

    var ch = choreo(prev, name);
    beginColour(ch.exhale ? EXHALE : STATES[name]);
    transitionUntil = now + ch.ms;
    applyState(name, ch);

    if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }
    holdUntil = 0;
    if (HOLD[name]) {
      holdUntil = now + HOLD[name];
      holdTimer = setTimeout(function () {
        holdTimer = null;
        holdUntil = 0;
        if (pending && pending !== stateId) { var n = pending; pending = null; setState(n); }
      }, HOLD[name] + 10);
    }

    // The exhale is a two-beat move: flare, then settle onto the resting green. The second
    // beat is scheduled rather than sprung, because it has to happen AFTER the flare peaks.
    if (ch.exhale) {
      setTimeout(function () {
        if (stateId !== 'DONE') return;
        var p = STATES.DONE;
        springs.intensity.target = p.intensity;
        springs.spec.target = p.spec;
        springs.scale.target = 1.0;
        beginColour(STATES.DONE);
        if (wake) wake();
      }, ch.exhale);
    }

    if (canvas && canvas.classList.contains('fb')) paintFallback();
    if (wake) wake();
  }

  function impulse(kind) {
    var now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    if (kind === 'token') {
      // Tokens arrive ~51 a turn. Un-throttled, every one of them would re-target the
      // pulse spring and the orb would buzz rather than breathe.
      if (now - lastToken < 70) return;
      lastToken = now;
      impToken = Math.min(1.0, impToken + 0.25);
      springs.pulse.target = Math.min(0.35, springs.pulse.target + 0.03);
    } else if (kind === 'tool') {
      impTool = 1.0;
      impRipple = Math.max(impRipple, 1.0);
      springs.energy.target = Math.min(1.15, springs.energy.value + 0.18);
      springs.pulse.target = Math.min(0.45, springs.pulse.target + 0.10);
      springs.scale.target = 1.03;
      setTimeout(function () {
        var p = STATES[stateId] || STATES.IDLE;
        springs.energy.target = p.energy;
        springs.pulse.target = p.pulse;
        springs.scale.target = 1.0;
        if (wake) wake();
      }, 420);
    }
    if (wake) wake();
  }

  function settled() {
    if (impToken > 0.02 || impTool > 0.02 || impRipple > 0.02) return false;
    var now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    if (now < transitionUntil) return false;
    if (!mix.settled(0.002)) return false;
    for (var key in springs) if (!springs[key].settled()) return false;
    return true;
  }

  function currentColour() { return lerpOklch(from, to, mix.value); }

  function paintFallback() {
    if (!canvas) return;
    canvas.style.setProperty('--orb-tint', oklchToCss(currentColour()));
  }

  // -- background luminance -------------------------------------------------
  // The shader adapts its halo and its tonemap white point to how bright the editor is
  // behind it. Read from #head, which is the element actually carrying the sidebar colour;
  // body is transparent here.
  function readBgLuma() {
    try {
      var host = document.getElementById('head') || document.body;
      var m = getComputedStyle(host).backgroundColor.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
      if (!m) return bgLuma;
      var f = function (x) { x = +x / 255; return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); };
      return 0.2126 * f(m[1]) + 0.7152 * f(m[2]) + 0.0722 * f(m[3]);
    } catch (e) { return bgLuma; }
  }

  // -- GL -------------------------------------------------------------------
  var VS = '#version 300 es\\nin vec2 p;void main(){gl_Position=vec4(p,0.0,1.0);}';
  var FS = ${JSON.stringify(FRAGMENT_SHADER)};

  function compile(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('orb: shader compile failed', gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  function start(el) {
    var gl = el.getContext('webgl2', {
      alpha: true, premultipliedAlpha: true, antialias: false, powerPreference: 'low-power'
    });
    if (!gl) return false;
    try { if ('drawingBufferColorSpace' in gl) gl.drawingBufferColorSpace = 'display-p3'; } catch (e) {}

    var vs = compile(gl, gl.VERTEX_SHADER, VS);
    var fs = compile(gl, gl.FRAGMENT_SHADER, FS);
    if (!vs || !fs) return false;

    var prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('orb: link failed', gl.getProgramInfoLog(prog));
      return false;
    }

    var vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    var u = {};
    ['u_res','u_time','u_flow','u_breath','u_intensity','u_pulse','u_spec','u_energy',
     'u_scale','u_ior','u_color','u_accent','u_bg','u_ripple','u_impulse'
    ].forEach(function (n) { u[n] = gl.getUniformLocation(prog, n); });

    var running = false, raf = 0, prev = 0, drawnAt = 0;

    function frame(now) {
      raf = 0;
      if (!document.body.contains(el)) { running = false; return; }
      if (document.hidden) { running = false; return; }
      if (el.offsetParent === null) { raf = requestAnimationFrame(frame); return; }

      // At rest the orb is only breathing, and a slow sine is indistinguishable at 30fps.
      // Halving the idle frame rate halves what a sidebar costs while nothing is happening.
      var quiet = settled() && stateId === 'IDLE';
      if (quiet && now - drawnAt < 32) { raf = requestAnimationFrame(frame); return; }
      drawnAt = now;

      var dt = frozen ? 0 : Math.min(Math.max(0, (now - prev) / 1000), 0.05);
      prev = now;

      impToken  = Math.max(0, impToken  - dt * 2.8);
      impTool   = Math.max(0, impTool   - dt * 1.6);
      impRipple = Math.max(0, impRipple - dt * 1.4);

      springs.intensity.update(dt);
      springs.flow.update(dt);
      springs.breath.update(dt);
      springs.pulse.update(dt);
      springs.spec.update(dt);
      springs.energy.update(dt);
      springs.scale.update(dt);
      mix.update(dt);

      // Integrate, do not multiply: the rate is what the state changes, so the PHASE has to
      // stay continuous across the change or the whole noise field jumps.
      animTime    += dt;
      flowPhase   += dt * springs.flow.value;
      breathPhase += dt * springs.breath.value;

      var c = currentColour();
      var lin = oklchToLinear(c), acc = oklchToLinear(accentOf(c));

      // Supersampled a little past the device ratio: the key highlight and the dispersion
      // fringe are sub-pixel features at this size, and the silhouette AA is analytic, so
      // the extra samples all go into detail rather than edges.
      var dpr = Math.min((window.devicePixelRatio || 1) * 1.5, 3);
      var w = Math.max(1, Math.floor(el.clientWidth * dpr));
      var h = Math.max(1, Math.floor(el.clientHeight * dpr));
      if (el.width !== w || el.height !== h) { el.width = w; el.height = h; gl.viewport(0, 0, w, h); }

      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(prog);
      gl.bindVertexArray(vao);

      gl.uniform2f(u.u_res, w, h);
      gl.uniform1f(u.u_time, animTime);
      gl.uniform1f(u.u_flow, flowPhase);
      gl.uniform1f(u.u_breath, breathPhase);
      gl.uniform1f(u.u_intensity, springs.intensity.value);
      gl.uniform1f(u.u_pulse, springs.pulse.value);
      gl.uniform1f(u.u_spec, springs.spec.value);
      gl.uniform1f(u.u_energy, springs.energy.value);
      gl.uniform1f(u.u_scale, springs.scale.value);
      gl.uniform1f(u.u_ior, 1.45);
      gl.uniform3f(u.u_color, lin[0], lin[1], lin[2]);
      gl.uniform3f(u.u_accent, acc[0], acc[1], acc[2]);
      gl.uniform1f(u.u_bg, bgLuma);
      gl.uniform1f(u.u_ripple, impRipple);
      gl.uniform1f(u.u_impulse, Math.max(impToken, impTool));

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      if (frozen) { running = false; return; }
      raf = requestAnimationFrame(frame);
    }

    wake = function () {
      if (document.hidden || running) return;
      running = true;
      prev = (typeof performance !== 'undefined' ? performance.now() : Date.now());
      raf = requestAnimationFrame(frame);
    };
    wake();
    return true;
  }

  // -- mount ----------------------------------------------------------------
  function mount() {
    canvas = document.getElementById('orb');
    if (!canvas) return;
    bgLuma = readBgLuma();

    var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced || !start(canvas)) {
      canvas.classList.add('fb');
      paintFallback();
    }

    document.addEventListener('visibilitychange', function () { if (!document.hidden && wake) wake(); });
    // The editor can switch theme under a live webview; the halo and the tonemap white
    // point both depend on how bright the background is, so re-read it when it changes.
    try {
      new MutationObserver(function () {
        bgLuma = readBgLuma();
        if (wake) wake();
      }).observe(document.body, { attributes: true, attributeFilter: ['class', 'style'] });
    } catch (e) {}
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();

  window.__orb = {
    state: setState,
    impulse: impulse,

    // Still-capture hook for scripts/preview-orb.js, and nothing in the extension calls it.
    // Jumps every spring onto its target, pins the motion phases, and draws exactly one
    // frame. Without it a screenshotter races the animation: the same command produced a
    // settled amber bead one run and a half-transitioned blue one the next, which makes
    // comparing two shots meaningless.
    snap: function (phase) {
      for (var k in springs) { springs[k].value = springs[k].target; springs[k].velocity = 0; }
      mix.value = 1; mix.velocity = 0;
      from = to;
      impToken = 0; impTool = 0; impRipple = 0;
      transitionUntil = 0; holdUntil = 0;
      if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }
      animTime = phase; flowPhase = phase * 0.4; breathPhase = phase * 0.9;
      frozen = true;
      if (wake) wake();
    },

    read: function () {
      return { state: stateId, colour: currentColour(), energy: springs.energy.value,
               intensity: springs.intensity.value, settled: settled(),
               fallback: !!(canvas && canvas.classList.contains('fb')) };
    }
  };
})();
`;
}
