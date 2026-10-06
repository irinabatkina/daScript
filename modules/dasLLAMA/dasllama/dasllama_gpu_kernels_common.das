options gen2
//! The kernels both GPU homes run, written once: a `class template` here carries a kernel's buffers, its
//! argument struct and its body, and each home stamps it under its own dispatch name.
module dasllama_gpu_kernels_common shared public
require dasllama/dasllama_lint public
require daslib/typemacro_boost
require math

struct GkAxpyArgs {
    nelem : uint
    scale : float
    zero : uint    //! 1 = o's old contents ignored
}

//! o = (zero ? 0 : o) + scale * y over nelem elements, one element an invocation
[ |> template_struct_instance]
class template GkAxpy {
    @ssbo @binding = 0 @role = "readwrite" o : array<float>
    @ssbo @binding = 1 @role = "read" y : array<float>
    @kargs ka : GkAxpyArgs

    def axpy_body {
        let i = gl_GlobalInvocationID.x
        if (i < ka.nelem) {
            let base = ka.zero != 0u ? 0.0 : o[i]
            o[i] = base + ka.scale * y[i]
        }
    }
}

struct GkRopeTabArgs {
    rows : uint   //! the rows roped, row r at the tables' row r past their element bases
    qs : uint     //! x's row stride
    col : uint    //! the roped span's first column (a fused [q | k | v] row's q or k slot)
    d : uint      //! the span's width, its heads side by side
    dh : uint     //! the head size
    coff : uint   //! the cos table [pos][dh / 2]'s element base in its binding
    soff : uint   //! the sin table's element base in its binding, laid out as the cos table
}

//! the rotary embedding over a span of rows in place from per-position tables: row r turns each pair of each head by
//! the tables' angle j at row r - NEOX the pairs (j, j + dh / 2), the CPU `rope_neox_tab_rows` (the vision mrope tables),
//! else the adjacent pairs (2j, 2j + 1), the CPU `rope_rows`; one pair an invocation over rows * d / 2
[ |> template_struct_instance]
class template GkRopeTab {
    @template_constant NEOX : bool = false
    @ssbo @binding = 0 @role = "readwrite" x : array<float>
    @ssbo @binding = 1 @role = "read" cos_t : array<float>
    @ssbo @binding = 2 @role = "read" sin_t : array<float>
    @kargs pa : GkRopeTabArgs

    def pair_first(hb, j : uint) : uint {
        static_if (NEOX) {
            return hb + j
        } else {
            return hb + 2u * j
        }
    }

    def pair_partner_dist(hh : uint) : uint {
        static_if (NEOX) {
            return hh
        } else {
            return 1u
        }
    }

    def rope_tab_body {
        let gid = gl_GlobalInvocationID.x
        let half = pa.d / 2u
        if (gid < pa.rows * half) {
            let r = gid / max(half, 1u)
            let pi = gid % max(half, 1u)
            let hh = max(pa.dh / 2u, 1u)
            let j = pi % hh
            let i0 = pair_first(r * pa.qs + pa.col + (pi / hh) * pa.dh, j)
            let i1 = i0 + pair_partner_dist(hh)
            let tb = r * hh + j
            let cv = cos_t[pa.coff + tb]
            let sv = sin_t[pa.soff + tb]
            let v0 = x[i0]
            let v1 = x[i1]
            x[i0] = v0 * cv - v1 * sv
            x[i1] = v0 * sv + v1 * cv
        }
    }
}

struct GkConcatArgs {
    c : uint          //! x's row width
    rc : uint         //! the residual columns, read only under `with_res`
    rs : uint         //! the residual rows' stride: 0 broadcasts one row to every position
    f0s : uint        //! the F0 column's row stride
    ns : uint         //! the noise column's row stride
    with_res : uint   //! 1 = the residual columns follow x
    width : uint      //! the output row width, the columns past the last input zero
    rows : uint
}

//! o [rows][width] = cat(x [rows][c], res at stride rs under `with_res`, f0 at stride f0s, nd at stride ns), zero-padded to
//! width - the CPU `concat_rows`; one element an invocation
[ |> template_struct_instance]
class template GkConcat {
    @ssbo @binding = 0 @role = "read" x : array<float>
    @ssbo @binding = 1 @role = "read" res : array<float>
    @ssbo @binding = 2 @role = "read" f0 : array<float>
    @ssbo @binding = 3 @role = "read" nd : array<float>
    @ssbo @binding = 4 @role = "write" o : array<float>
    @kargs ka : GkConcatArgs

    def concat_body {
        let gid = gl_GlobalInvocationID.x
        if (gid < ka.rows * ka.width) {
            let w = max(ka.width, 1u)
            let r = gid / w
            let j = gid % w
            let rc = ka.with_res != 0u ? ka.rc : 0u
            var v = 0.0
            if (j < ka.c) {
                v = x[r * ka.c + j]
            } elif (j < ka.c + rc) {
                v = res[r * ka.rs + (j - ka.c)]
            } elif (j == ka.c + rc) {
                v = f0[r * ka.f0s]
            } elif (j == ka.c + rc + 1u) {
                v = nd[r * ka.ns]
            }
            o[gid] = v
        }
    }
}

struct GkPkRowsArgs {
    rows : uint
    width : uint   //! the floats a row copies
    ss : uint      //! src's row stride
    sc : uint      //! src's first column
    sr0 : uint     //! src's first row
    ds : uint      //! dst's row stride
    dc : uint      //! dst's first column
    dr0 : uint     //! dst's first row
}

//! `rows` rows of `width` floats from src [.][ss] at column sc, rows sr0.., into dst [.][ds] at column dc, rows dr0.. - the
//! Pocket chains' row copy (a conv's window behind its carry, a carry's tail to its head, the K and V columns into their
//! caches, the sample column out); the ELU stamp maps each value on the way (the CPU `elu_rows` before a conv); one element an invocation
[ |> template_struct_instance]
class template GkPkRows {
    @template_constant ELU : bool = false
    @ssbo @binding = 0 @role = "read" src : array<float>
    @ssbo @binding = 1 @role = "write" dst : array<float>
    @kargs pa : GkPkRowsArgs

    def pk_rows_body {
        let gid = gl_GlobalInvocationID.x
        if (gid < pa.rows * pa.width) {
            let w = max(pa.width, 1u)
            let r = gid / w
            let j = gid % w
            var v = src[(pa.sr0 + r) * pa.ss + pa.sc + j]
            static_if (ELU) {
                v = max(v, 0.0) + min(exp(v) - 1.0, 0.0)
            }
            dst[(pa.dr0 + r) * pa.ds + pa.dc + j] = v
        }
    }
}

struct GkSigSumArgs {
    t : uint       //! the rows
    nb : uint      //! the logits summed a row
    ls : uint      //! the logit rows' stride
    speed : float  //! the divisor of each row's sum
}

//! raw[r] = (the sum of sigmoid(logits[r][j]) over j < nb, in j order) / speed - the CPU `styletts2_durations`' raw
//! durations (the CPU divides; a reciprocal multiply rounds a raw duration apart from it); one row an invocation
[ |> template_struct_instance]
class template GkSigSum {
    @ssbo @binding = 0 @role = "read" logits : array<float>
    @ssbo @binding = 1 @role = "write" raw : array<float>
    @kargs pa : GkSigSumArgs

    def sigsum_body {
        let r = gl_GlobalInvocationID.x
        if (r < pa.t) {
            var acc = 0.0
            var j = 0u
            while (j < pa.nb) {
                acc += sigmoid_f32(logits[r * pa.ls + j])
                j++
            }
            raw[r] = acc / pa.speed
        }
    }
}

struct GkSourceArgs {
    n : uint            //! the samples, frames * up
    nlow : uint         //! the phase frames, `resize_len(n, 1 / up)`
    nh : uint           //! the harmonics
    up : uint
    sr : float
    sine_amp : float
    noise_std : float
    voiced_thr : float
    seed : uint         //! the own noise draw's key
    woff : uint         //! the source linear's weight row [nh] in the slab
    boff : uint         //! its bias [1] in the slab
}

//! the phase per frame lowc [nh][nlow] as the running sum of the increments - TORCH: the double accumulator as a two-float
//! sum carrying its rounding error, else the plain float sum; one workgroup, invocation h < nh walks its harmonic
[ |> template_struct_instance]
class template GkSrcCumsum {
    @template_constant TORCH : bool = false
    @ssbo @binding = 0 @role = "read" low : array<float>
    @ssbo @binding = 1 @role = "write" lowc : array<float>
    @kargs pa : GkSourceArgs

    def src_cumsum_body {
        let h = gl_LocalInvocationID.x
        if (h < pa.nh) {
            var acc = 0.0
            var lo = 0.0
            var k = 0u
            while (k < pa.nlow) {
                let x = low[h * pa.nlow + k]
                static_if (TORCH) {
                    tts_two_sum_add(acc, lo, x)
                } else {
                    acc += x
                }
                lowc[h * pa.nlow + k] = tts_phase_rad(acc, pa.up)
                k++
            }
        }
    }
}

//! the source's own noise draw noise_n [n][nh], `tts_hash_normal` at the seed - the captured stream's seat when no stream
//! was captured; one element an invocation
[ |> template_struct_instance]
class template GkSrcNoise {
    @ssbo @binding = 0 @role = "write" noise_n : array<float>
    @kargs pa : GkSourceArgs

    def src_noise_body {
        let gid = gl_GlobalInvocationID.x
        if (gid < pa.n * pa.nh) {
            noise_n[gid] = tts_hash_normal(pa.seed, gid)
        }
    }
}

struct GkStftArgs {
    n : uint        //! the signal's samples
    frames : uint
    bins : uint
    k : uint        //! the frame length, each basis row's taps
    hop : uint
    pad : uint      //! the samples padded on each side
    eps : float
    reoff : uint    //! the real basis wre [bins][k] in wn
    imoff : uint    //! the imaginary basis, the same layout
}

//! the STFT as two convs over the padded signal - REFLECT mirrors the signal past its ends, else the edge sample repeats -
//! into spec [frames][2 bins], the magnitudes sqrt(re^2 + im^2 + eps) then the phases, a zero imaginary part on the
//! negative real axis reading +pi (the CPU `magnitude_phase`); one (frame, bin) an invocation
[ |> template_struct_instance]
class template GkStft {
    @template_constant REFLECT : bool = false
    @ssbo @binding = 0 @role = "read" har : array<float>
    @ssbo @binding = 1 @role = "weight" wn : array<float>
    @ssbo @binding = 2 @role = "write" spec : array<float>
    @kargs pa : GkStftArgs

    def stft_body {
        let gid = gl_GlobalInvocationID.x
        let bins = max(pa.bins, 1u)
        if (gid < pa.frames * bins) {
            let f = gid / bins
            let b = gid % bins
            var re = 0.0
            var im = 0.0
            var tap = 0u
            while (tap < pa.k) {
                let p = int(f * pa.hop + tap) - int(pa.pad)
                var src = 0
                static_if (REFLECT) {
                    src = tts_pad_reflect(p, int(pa.n))
                } else {
                    src = clamp(p, 0, int(pa.n) - 1)
                }
                let v = har[uint(src)]
                re += wn[pa.reoff + b * pa.k + tap] * v
                im += wn[pa.imoff + b * pa.k + tap] * v
                tap++
            }
            let row = f * 2u * pa.bins
            spec[row + b] = tts_stft_mag(re, im, pa.eps)
            spec[row + pa.bins + b] = tts_stft_phase(re, im)
        }
    }
}

struct GkIstftArgs {
    tt : uint         //! the frames
    stride : uint     //! y's row width
    bins : uint
    k : uint          //! the frame length
    hop : uint
    pad : uint        //! the samples trimmed from each end
    envelope : uint   //! 1 = the summed squared window divided out where it is not negligible
    n : uint          //! the samples out
    reoff : uint      //! the real basis wre [k][bins] in wn - the transposed conv's operand layout
    imoff : uint      //! the imaginary basis, the same layout
    wnoff : uint      //! the window [k] in wn
}

//! the inverse STFT from conv_post's rows y [tt][stride] - a bin's log magnitude, then its phase - as the overlap-added
//! transposed conv, a non-negligible envelope divided out under `envelope`; one sample an invocation
[ |> template_struct_instance]
class template GkIstft {
    @ssbo @binding = 0 @role = "read" y : array<float>
    @ssbo @binding = 1 @role = "weight" wn : array<float>
    @ssbo @binding = 2 @role = "write" wave : array<float>
    @kargs pa : GkIstftArgs

    def istft_body {
        let i = gl_GlobalInvocationID.x
        if (i < pa.n) {
            let m = i + pa.pad
            let hop = max(pa.hop, 1u)
            var acc = 0.0
            var env = 0.0
            var tap = 0u
            while (tap < pa.k) {
                if (m >= tap && (m - tap) % hop == 0u) {
                    let q = (m - tap) / hop
                    if (q < pa.tt) {
                        var b = 0u
                        while (b < pa.bins) {
                            let wa = tap * pa.bins + b
                            acc += tts_istft_term(y[q * pa.stride + b], y[q * pa.stride + pa.bins + b], wn[pa.reoff + wa], wn[pa.imoff + wa])
                            b++
                        }
                        let wv = wn[pa.wnoff + tap]
                        env += wv * wv
                    }
                }
                tap++
            }
            wave[i] = (pa.envelope != 0u && env > 1e-11) ? acc / env : acc
        }
    }
}

struct GkAddScaleArgs {
    nelem : uint
    scale : float
}

//! o = (a + b) * scale over nelem elements - the residual join's 1 / sqrt(2) and the averaged stage sums; o may be a;
//! one element an invocation
[ |> template_struct_instance]
class template GkAddScale {
    @ssbo @binding = 0 @role = "read" a : array<float>
    @ssbo @binding = 1 @role = "read" b : array<float>
    @ssbo @binding = 2 @role = "write" o : array<float>
    @kargs pa : GkAddScaleArgs

    def add_scale_body {
        let i = gl_GlobalInvocationID.x
        if (i < pa.nelem) {
            o[i] = (a[i] + b[i]) * pa.scale
        }
    }
}

//! the source's resample law - TORCH: torch's taps and mix, else onnxruntime's - the base of the source stamps; the taps'
//! quotients divide through `tts_div`
[ |> template_struct_instance]
class template GkSrcLaw {
    @template_constant TORCH : bool = false

    def law_taps(n, to : uint; scale : float; var i0, i1 : uint&; var l0, l1 : float&) {
        static_if (TORCH) {
            tts_resize_taps_torch(n, to, tts_div(1.0, scale), i0, i1, l0, l1)
        } else {
            tts_resize_taps_onnx(n, tts_div(float(to) + 0.5, scale), i0, i1, l0, l1)
        }
    }

    def law_mix(x0, x1, l0, l1 : float; last : bool) : float {
        static_if (TORCH) {
            return tts_resize_mix_torch(x0, x1, l0, l1)
        } else {
            return tts_resize_mix_onnx(x0, x1, l0, l1, last)
        }
    }
}

//! the low-rate phase increments low [nh][nlow] on the stamp's resample law, the first sample's initial phases folded in;
//! one element an invocation
[ |> template_struct_instance]
class template GkSrcLow : GkSrcLaw {
    @ssbo @binding = 0 @role = "read" f0 : array<float>
    @ssbo @binding = 1 @role = "read" noise_u : array<float>
    @ssbo @binding = 2 @role = "write" low : array<float>
    @kargs pa : GkSourceArgs

    def src_low_body {
        let gid = gl_GlobalInvocationID.x
        if (gid < pa.nh * pa.nlow) {
            let nlow = max(pa.nlow, 1u)
            let up = max(pa.up, 1u)
            let h = gid / nlow
            var i0 = 0u
            var i1 = 0u
            var l0 = 0.0
            var l1 = 0.0
            law_taps(pa.n, gid % nlow, tts_div(1.0, float(pa.up)), i0, i1, l0, l1)
            let r0 = tts_src_rad(tts_div(f0[i0 / up] * float(h + 1u), pa.sr), noise_u[h], i0, h)
            let r1 = tts_src_rad(tts_div(f0[i1 / up] * float(h + 1u), pa.sr), noise_u[h], i1, h)
            low[gid] = law_mix(r0, r1, l0, l1, h + 1u == pa.nh)
        }
    }
}

//! the mixed source signal har [n] on the stamp's resample law: the harmonics' sines at the interpolated phases, the noise
//! rows scaled by voicing, the source linear (its weight row [nh] at woff, its bias [1] at boff in wn) and its tanh; one
//! sample an invocation
[ |> template_struct_instance]
class template GkSrcSines : GkSrcLaw {
    @ssbo @binding = 0 @role = "read" f0 : array<float>
    @ssbo @binding = 1 @role = "read" lowc : array<float>
    @ssbo @binding = 2 @role = "read" noise_n : array<float>
    @ssbo @binding = 3 @role = "weight" wn : array<float>
    @ssbo @binding = 4 @role = "write" har : array<float>
    @kargs pa : GkSourceArgs

    def src_sines_body {
        let i = gl_GlobalInvocationID.x
        if (i < pa.n) {
            var i0 = 0u
            var i1 = 0u
            var l0 = 0.0
            var l1 = 0.0
            law_taps(pa.nlow, i, float(pa.up), i0, i1, l0, l1)
            let uv = f0[i / max(pa.up, 1u)] > pa.voiced_thr ? 1.0 : 0.0
            let noise_amp = tts_noise_amp(uv, pa.noise_std, pa.sine_amp)
            var acc = wn[pa.boff]
            var h = 0u
            while (h < pa.nh) {
                let phase = law_mix(lowc[h * pa.nlow + i0], lowc[h * pa.nlow + i1], l0, l1, h + 1u == pa.nh)
                acc += tts_src_term(wn[pa.woff + h], tts_sin_big(phase) * pa.sine_amp, uv, noise_amp, noise_n[i * pa.nh + h])
                h++
            }
            har[i] = tts_tanh(acc)
        }
    }
}

let GK_ADAIN_EPS = 1e-5   //! the instance norms' eps - the body spells the literal (a kernel reads no module global); a driver holds it to its model's

struct GkAdainArgs {
    c : uint       //! the channels
    t : uint       //! the rows the statistics ran over
    hoff : uint    //! the style fc rows' base in h: gamma [c], then beta [c]
    gwoff : uint   //! the norm's scale row in wn
    gboff : uint   //! the norm's shift row in wn
    aoff : uint    //! Snake's alpha row in wn, read by the Snake stamp alone
}

//! AdaIN as the CPU `adain_affine` folds it - the column statistics, the norm's own scale and shift, then (1 + gamma) and
//! beta - followed by Snake or LeakyReLU(0.2); y may be x itself; one element an invocation over t * c
[ |> template_struct_instance]
class template GkAdain {
    @template_constant SNAKE : bool = false
    @ssbo @binding = 0 @role = "read" x : array<float>
    @ssbo @binding = 1 @role = "write" y : array<float>
    @ssbo @binding = 2 @role = "read" stats : array<float>
    @ssbo @binding = 3 @role = "read" h : array<float>
    @ssbo @binding = 4 @role = "weight" wn : array<float>
    @kargs pa : GkAdainArgs

    def adain_body {
        let gid = gl_GlobalInvocationID.x
        if (gid < pa.t * pa.c) {
            let ci = gid % max(pa.c, 1u)
            var scale = 0.0
            var shift = 0.0
            tts_adain_affine(stats[ci], stats[pa.c + ci], float(pa.t), 1e-5, wn[pa.gwoff + ci], wn[pa.gboff + ci], h[pa.hoff + ci],
                h[pa.hoff + pa.c + ci], scale, shift)
            let v = mad(x[gid], scale, shift)
            static_if (SNAKE) {
                let a = wn[pa.aoff + ci]
                let sn = sin(a * v)
                y[gid] = mad(1.0 / a, sn * sn, v)
            } else {
                y[gid] = v > 0.0 ? v : v * 0.2
            }
        }
    }
}

struct GkDwConvRowsArgs {
    c : uint       //! the channels, one tap row each
    k : uint       //! the taps, where the stamp does not fix them
    stride : uint  //! the transposed stamp's stride
    pad_l : uint   //! the transposed stamp's cropped left edge
    dil : uint     //! the transposed stamp's dilation
    t_in : uint    //! the input rows, and the output rows of the causal and the centered stamps
    t_out : uint   //! the transposed stamp's output rows
    woff : uint    //! w [c][k]'s element base in wn
    boff : uint    //! the transposed stamp's bias row in wn
    bnwoff : uint  //! the centered stamp's folded BatchNorm scale row in wn
    bnboff : uint  //! the centered stamp's folded BatchNorm shift row in wn
}

//! a depthwise conv on rows, y [t_out][c] from x [t_in][c] and w [c][k] (w [k][c] under TAP_MAJOR, parakeet's load-time repack), one
//! element an invocation; FORM is the source rule and the epilogue: 0 the transposed conv gathered, bias first and one mad a tap (the CPU
//! `conv1d_rows_transposed_depthwise`); 1 the causal conv, a row reading the k - 1 rows before it (gemma4a); 2 the centered conv, folded BatchNorm and silu after it
[ |> template_struct_instance]
class template GkDwConvRows {
    @template_constant FORM : int = 0   //!< spelled as its literal: a template constant folds only a literal
    @template_constant K : uint = 0u    //!< the taps where the stamp fixes them, else 0 and the taps are `pa.k`
    @template_constant TAP_MAJOR : bool = false
    @ssbo @binding = 0 @role = "read" x : array<float>
    @ssbo @binding = 1 @role = "write" y : array<float>
    @ssbo @binding = 2 @role = "weight" wn : array<float>
    @kargs pa : GkDwConvRowsArgs

    def rows_out : uint {
        static_if (FORM == 0) {
            return pa.t_out
        } else {
            return pa.t_in
        }
    }

    def taps : uint {
        static_if (K != 0u) {
            return K
        } else {
            return pa.k
        }
    }

    def src_row(t, kk : uint) : int {
        static_if (FORM == 0) {
            return conv_src_tr_row(t, kk, 0u, pa.pad_l, pa.dil, max(pa.stride, 1u), pa.t_in)
        } else {
            static_if (FORM == 1) {
                return conv_src_fwd_row(t, kk, 0u, taps() - 1u, 1u, 1u, pa.t_in)
            } else {
                return conv_src_fwd_row(t, kk, 0u, (taps() - 1u) / 2u, 1u, 1u, pa.t_in)
            }
        }
    }

    def tap(j, kk, nk : uint) : float {
        static_if (TAP_MAJOR) {
            return wn[pa.woff + kk * pa.c + j]
        } else {
            return wn[pa.woff + j * nk + kk]
        }
    }

    def dw_conv_rows_body {
        let e = gl_GlobalInvocationID.x
        if (e < rows_out() * pa.c) {
            let c = max(pa.c, 1u)
            let t = e / c
            let j = e % c
            let nk = taps()
            var acc = 0.0
            static_if (FORM == 0) {
                acc = wn[pa.boff + j]
            }
            var kk = 0u
            while (kk < nk) {
                let src = src_row(t, kk)
                if (src >= 0) {
                    static_if (FORM == 0) {
                        acc = mad(tap(j, kk, nk), x[uint(src) * pa.c + j], acc)
                    } else {
                        acc += tap(j, kk, nk) * x[uint(src) * pa.c + j]
                    }
                }
                kk++
            }
            static_if (FORM == 2) {
                y[e] = silu_f32(acc * wn[pa.bnwoff + j] + wn[pa.bnboff + j])
            } else {
                y[e] = acc
            }
        }
    }
}

//! the workgroup reductions every row kernel of either home shares: one barrier a reduce, every thread summing the subgroup
//! partials in the same order, so every thread holds the same sum or max; the lane primitives are each home's
//! `gk_subgroup_add` / `gk_subgroup_max`, and the result lands by reference, the statement form the MSL emitter splices
[ |> template_struct_instance]
class template GkWgReduce {
    @workgroup part : float[128]   //! two slots of 64 subgroup partials, one slot a reduce

    //! the workgroup sum every thread holds alike; `slot` alternates 0 / 1 between consecutive reduces; a statement with its
    //! result by reference, the form the MSL emitter inlines
    def wg_sum_into(v0 : float; slot : uint; var tot : float&) {
        let ss = gk_subgroup_add(v0)
        if (gl_SubgroupInvocationID == 0u) {
            part[slot * 64u + gl_SubgroupID] = ss
        }
        barrier()
        tot = 0.0
        var sgi = 0u
        while (sgi < gl_NumSubgroups) {
            tot += part[slot * 64u + sgi]
            sgi++
        }
    }

    //! the workgroup max every thread holds alike
    def wg_max_into(v0 : float; slot : uint; var m : float&) {
        let sm = gk_subgroup_max(v0)
        if (gl_SubgroupInvocationID == 0u) {
            part[slot * 64u + gl_SubgroupID] = sm
        }
        barrier()
        m = -3.0e38
        var sgi = 0u
        while (sgi < gl_NumSubgroups) {
            m = max(m, part[slot * 64u + sgi])
            sgi++
        }
    }
}

let GK_PK_ATTN_HS = 64u            //! the head size the attention row is stamped for, the body's literal: another head size declines at the driver
let GK_PK_ATTN_MAX_KEYS = 2048u    //! the scores a row stages, the body's literal: a longer key range declines at the driver

struct GkPkAttnArgs {
    t : uint      //! the query rows
    qs : uint     //! q's row stride
    qcol : uint   //! the query span's first column in q's rows
    d : uint      //! the heads' width, the cache rows' and o's row stride
    pos0 : uint   //! the first query's position
    ctx : uint    //! the keys a query sees at most, 0 = every key up to itself
}

//! the causal cached attention of one query row over the caches' rows (the CPU `attention_causal_rows`): query i at position
//! pos0 + i sees the keys up to itself, the last `ctx` of them when ctx is nonzero; the scaled query staged, the scores staged
//! and softmaxed in two passes, the value sum in four parts of 64 columns; a workgroup of 256 a (head, row), reductions as statement methods
[ |> template_struct_instance]
class template GkPkAttn : GkWgReduce {
    @ssbo @binding = 0 @role = "read" q : array<float>
    @ssbo @binding = 1 @role = "read" k : array<float>
    @ssbo @binding = 2 @role = "read" v : array<float>
    @ssbo @binding = 3 @role = "write" o : array<float>
    @kargs pa : GkPkAttnArgs
    @workgroup qh : float[64]
    @workgroup sc : float[2048]   //! == GK_PK_ATTN_MAX_KEYS, the array's literal
    @workgroup po : float[256]
    //! a column's quarter of the value sum: the keys `quarter`, `quarter` + 4, ... in order, eight a step with their loads
    //! in flight together, the column at `vb` in the first key's row
    def value_sum_into(vb, quarter, nk : uint; var acc : float&) {
        acc = 0.0
        var jj = quarter
        while (jj + 28u < nk) {
            let v0 = v[vb + jj * pa.d]
            let v1 = v[vb + (jj + 4u) * pa.d]
            let v2 = v[vb + (jj + 8u) * pa.d]
            let v3 = v[vb + (jj + 12u) * pa.d]
            let v4 = v[vb + (jj + 16u) * pa.d]
            let v5 = v[vb + (jj + 20u) * pa.d]
            let v6 = v[vb + (jj + 24u) * pa.d]
            let v7 = v[vb + (jj + 28u) * pa.d]
            acc += sc[jj] * v0
            acc += sc[jj + 4u] * v1
            acc += sc[jj + 8u] * v2
            acc += sc[jj + 12u] * v3
            acc += sc[jj + 16u] * v4
            acc += sc[jj + 20u] * v5
            acc += sc[jj + 24u] * v6
            acc += sc[jj + 28u] * v7
            jj += 32u
        }
        while (jj < nk) {
            acc += sc[jj] * v[vb + jj * pa.d]
            jj += 4u
        }
    }

    def pk_attn_body {
        let wg = gl_WorkGroupID.x
        let t = max(pa.t, 1u)
        let h = wg / t
        let i = wg % t
        let lid = gl_LocalInvocationID.x
        let p = pa.pos0 + i
        let j0 = tts_window_start(p, pa.ctx)
        let nk = p + 1u - j0
        if (lid < 64u) {
            qh[lid] = q[i * pa.qs + pa.qcol + h * 64u + lid] / sqrt(float(64u))
        }
        barrier()
        var m = -1.0e30
        var j = lid
        while (j < nk) {
            let krow = (j0 + j) * pa.d + h * 64u
            var sdot = 0.0
            var e = 0u
            while (e < 64u) {
                let k0 = k[krow + e]
                let k1 = k[krow + e + 1u]
                let k2 = k[krow + e + 2u]
                let k3 = k[krow + e + 3u]
                sdot += qh[e] * k0
                sdot += qh[e + 1u] * k1
                sdot += qh[e + 2u] * k2
                sdot += qh[e + 3u] * k3
                e += 4u
            }
            sc[j] = sdot
            m = max(m, sdot)
            j += 256u
        }
        var mall = 0.0
        wg_max_into(m, 0u, mall)
        var l = 0.0
        j = lid
        while (j < nk) {
            let pw = exp(sc[j] - mall)
            sc[j] = pw
            l += pw
            j += 256u
        }
        var lsum = 0.0
        wg_sum_into(l, 1u, lsum)
        let inv = 1.0 / lsum
        j = lid
        while (j < nk) {
            sc[j] = sc[j] * inv
            j += 256u
        }
        barrier()
        let col = lid % 64u
        let quarter = lid / 64u
        var vs = 0.0
        value_sum_into(j0 * pa.d + h * 64u + col, quarter, nk, vs)
        po[quarter * 64u + col] = vs
        barrier()
        if (lid < 64u) {
            var tot = 0.0
            var pp = 0u
            while (pp < 4u) {
                tot += po[pp * 64u + lid]
                pp++
            }
            o[i * pa.d + h * 64u + lid] = tot
        }
    }
}

struct GkGatedRowsArgs {
    d : uint        //! the output row width; the input rows are 2 d wide, the gate in their second half
    nelem : uint    //! rows x d
}

//! the gated halves of a row, one element an invocation: FORM 0 the conv module's GLU, first . sigmoid(second) (gemma4a, the
//! FastConformers); FORM 1 the ultravox projector's swapped SwiGLU, silu(second) . first
[ |> template_struct_instance]
class template GkGatedRows {
    @template_constant FORM : int = 0
    @ssbo @binding = 0 @role = "read" h : array<float>
    @ssbo @binding = 1 @role = "write" y : array<float>
    @kargs pa : GkGatedRowsArgs

    def gated_rows_body {
        let e = gl_GlobalInvocationID.x
        if (e < pa.nelem) {
            let dd = max(1u, pa.d)
            let t = e / dd
            let c = e % dd
            let a = h[t * 2u * pa.d + c]
            let b = h[t * 2u * pa.d + pa.d + c]
            static_if (FORM == 0) {
                y[e] = sig_gate(a, b)
            } else {
                y[e] = silu_f32(b) * a
            }
        }
    }
}

struct GkLnReluRowsArgs {
    cout : uint    //! the channels a row normalizes over
    ldin : uint    //! x's row stride (the conv GEMM's padded width)
    ldout : uint   //! y's row stride
    rows : uint
    woff : uint    //! the weight row's base in wn
    eps : float
}

//! the subsample convs' LayerNorm over a row's channels (weight only, biased variance) then ReLU, one thread a row: y's compact
//! rows off x's padded ones, or in place at one stride (gemma4a's front on both homes)
[ |> template_struct_instance]
class template GkLnReluRows {
    @ssbo @binding = 0 @role = "read" x : array<float>
    @ssbo @binding = 1 @role = "write" y : array<float>
    @ssbo @binding = 2 @role = "weight" wn : array<float>
    @kargs pa : GkLnReluRowsArgs

    def ln_relu_rows_body {
        let row = gl_GlobalInvocationID.x
        if (row < pa.rows) {
            let base = row * pa.ldin
            let n = float(max(1u, pa.cout))
            var mean = 0.0
            for (c in urange(pa.cout)) {
                mean += x[base + c]
            }
            mean /= n
            var vv = 0.0
            for (c in urange(pa.cout)) {
                let dv = x[base + c] - mean
                vv += dv * dv
            }
            let inv = rsqrt(vv / n + pa.eps)
            let obase = row * pa.ldout
            for (c in urange(pa.cout)) {
                y[obase + c] = max((x[base + c] - mean) * inv * wn[pa.woff + c], 0.0)
            }
        }
    }
}

struct GkFeatShuffleArgs {
    w : uint         //! the frequency rows a time step keeps
    ch : uint        //! the channels
    t : uint         //! the time steps
    rows_pad : uint  //! the rows written: t, or past it the GEMM's zero row pad
}

//! the subsample stack's feature rows: out[t][f + w c] = in[(t w + f) ch + c] - the [time][freq][ch] image as [time][ch][freq] rows,
//! the rows past t zero
[ |> template_struct_instance]
class template GkFeatShuffle {
    @ssbo @binding = 0 @role = "read" src : array<float>
    @ssbo @binding = 1 @role = "write" dst : array<float>
    @kargs pa : GkFeatShuffleArgs

    def feat_shuffle_body {
        let gid = gl_GlobalInvocationID.x
        if (gid < pa.rows_pad * pa.w * pa.ch) {
            let wc = max(1u, pa.w * pa.ch)
            let w = max(1u, pa.w)
            let t = gid / wc
            if (t >= pa.t) {
                dst[gid] = 0.0
            } else {
                let rem = gid % wc
                let c = rem / w
                let f = rem % w
                dst[gid] = src[(t * pa.w + f) * pa.ch + c]
            }
        }
    }
}

struct GkRelSinCosArgs {
    d : uint         //! the row width, the (sin, cos) pairs interleaved
    tt : uint        //! the encoder rows: the table's 2 tt - 1 rows, row w carrying the relative position tt - 1 - w
    rows_pad : uint  //! the rows written: 2 tt - 1, or past it the GEMM's zero row pad
}

//! the Transformer-XL rel-position table [ww x d]: row w = the sinusoid of tt - 1 - w at the d / 2 frequencies, (sin, cos)
//! interleaved - the CPU `rel_pos_table`'s form; the rows past 2 tt - 1 zero
[ |> template_struct_instance]
class template GkRelSinCos {
    @ssbo @binding = 0 @role = "write" tab : array<float>
    @kargs pa : GkRelSinCosArgs

    def rel_sincos_body {
        let gid = gl_GlobalInvocationID.x
        let d = max(1u, pa.d)
        if (pa.tt > 0u && gid < pa.rows_pad * pa.d) {
            let w = gid / d
            if (w >= 2u * pa.tt - 1u) {
                tab[gid] = 0.0
            } else {
                let e = gid % d
                let k = e / 2u
                let rel = float(int(pa.tt) - 1 - int(w))
                let freq = exp(-(float(k * 2u) * 9.210340371976184 / float(d)))
                let theta = freq * rel
                tab[gid] = (e & 1u) == 0u ? sin(theta) : cos(theta)
            }
        }
    }
}

struct GkDwConv2dArgs {
    iw : uint        //! the source block's width, height and channels ([ih x iw x ch], c fastest)
    ih : uint
    ih_valid : uint  //! the rows a tap may read - the length mask: the rows past them add nothing
    ch : uint
    woff : uint    //! the taps [ch x 9] in wn, (kh 3 + kw), or [9 x ch] under TAP_MAJOR
    boff : uint    //! the bias row [ch] in wn
}

//! a depthwise conv2d k3 s2 p1 with its bias (the FastConformer subsample stacks), bias first, one thread an output element;
//! a tap off the image adds nothing
[ |> template_struct_instance]
class template GkDwConv2d {
    @template_constant TAP_MAJOR : bool = false
    @ssbo @binding = 0 @role = "read" x : array<float>
    @ssbo @binding = 1 @role = "write" y : array<float>
    @ssbo @binding = 2 @role = "weight" wn : array<float>
    @kargs pa : GkDwConv2dArgs

    def tap2d(c, k9 : uint) : float {
        static_if (TAP_MAJOR) {
            return wn[pa.woff + k9 * pa.ch + c]
        } else {
            return wn[pa.woff + c * 9u + k9]
        }
    }

    def dw_conv2d_body {
        let gid = gl_GlobalInvocationID.x
        let ch = max(1u, pa.ch)
        let ow = (pa.iw - 1u) / 2u + 1u
        let oh = (pa.ih - 1u) / 2u + 1u
        if (gid < oh * ow * pa.ch) {
            let c = gid % ch
            let o = gid / ch
            let oy = o / ow
            let ox = o % ow
            var acc = wn[pa.boff + c]
            var kh = 0u
            while (kh < 3u) {
                let iy = int(oy * 2u + kh) - 1
                if (iy >= 0 && iy < int(pa.ih_valid)) {
                    var kw = 0u
                    while (kw < 3u) {
                        let ix = int(ox * 2u + kw) - 1
                        if (ix >= 0 && ix < int(pa.iw)) {
                            acc += tap2d(c, kh * 3u + kw) * x[(uint(iy) * pa.iw + uint(ix)) * ch + c]
                        }
                        kw++
                    }
                }
                kh++
            }
            y[gid] = acc
        }
    }
}

let GK_G4A_ATTN_PAST = 12u   //! gemma4a's causal window - self and eleven past rows: `GkG4aAttn`'s PAST default spells it, each driver verifies the family's value against it
let GK_G4A_ATTN_CAP = 50.0   //! gemma4a's score soft cap, verified the same way

struct GkG4aAttnArgs {
    d : uint        //! the row width (compact [npos x d] rows)
    npos : uint
    hs : uint       //! the head width where the stamp leaves HS at 0
    pdsoff : uint   //! the per-dim query scale row's base in wn
    cap : float     //! the score soft cap
    qscale : float  //! the query scale ahead of the per-dim scale, (1 / sqrt(hs)) / log 2 - the CPU chain's log2-domain softmax
    kscale : float  //! the key scale, log(1 + e) / log 2
}

//! gemma4a's causal sliding-window attention with the layer's projected RPE table: one invocation a (query row, head), the
//! window sixteen rows at most (the score array; gemma4a's window is twelve), the head width the stamp's HS or `pa.hs`;
//! the query and key scales come from the host (`gk_g4a_attn_scales`), the one rounding of them both homes share
[ |> template_struct_instance]
class template GkG4aAttn {
    @template_constant HS : uint = 0u     //!< the head width served, 0 for `pa.hs`
    @template_constant PAST : uint = 12u  //!< the causal window: `GK_G4A_ATTN_PAST`, spelled as its literal
    @ssbo @binding = 0 @role = "read" q : array<float>
    @ssbo @binding = 1 @role = "read" k : array<float>
    @ssbo @binding = 2 @role = "read" v : array<float>
    @ssbo @binding = 3 @role = "write" ao : array<float>
    @ssbo @binding = 4 @role = "read" rpe : array<float>
    @ssbo @binding = 5 @role = "weight" wn : array<float>
    @kargs pa : GkG4aAttnArgs

    def head_width : uint {
        static_if (HS != 0u) {
            return HS
        } else {
            return pa.hs
        }
    }

    def g4a_attn_body {
        let gid = gl_GlobalInvocationID.x
        let npos = max(1u, pa.npos)
        let i = gid % npos
        let h = gid / npos
        let hs = head_width()
        if (h * hs >= pa.d) {
            return
        }
        let d = pa.d
        let hoff = h * hs
        let qscale = pa.qscale
        let kscale = pa.kscale
        let j0 = i >= PAST - 1u ? i - (PAST - 1u) : 0u
        var sarr : float[16]
        var m = -1.5e38
        var j = j0
        while (j <= i) {
            let rr = (PAST - (i - j)) * d
            var cs = 0.0
            var rs = 0.0
            var n = 0u
            while (n < hs) {
                let qv = q[i * d + hoff + n] * qscale * wn[pa.pdsoff + n]
                cs += qv * (k[j * d + hoff + n] * kscale)
                rs += qv * rpe[rr + hoff + n]
                n++
            }
            let sc = softcap_exp(cs + rs, pa.cap)
            sarr[j - j0] = sc
            m = max(m, sc)
            j++
        }
        var stot = 0.0
        j = j0
        while (j <= i) {
            let p = exp(sarr[j - j0] - m)
            sarr[j - j0] = p
            stot += p
            j++
        }
        let inv = 1.0 / stot
        var n = 0u
        while (n < hs) {
            var acc = 0.0
            j = j0
            while (j <= i) {
                acc += sarr[j - j0] * v[j * d + hoff + n]
                j++
            }
            ao[i * d + hoff + n] = acc * inv
            n++
        }
    }
}

struct GkRope2dArgs {
    nheads : uint
    hs : uint       //! the head width: the compact row is nheads x hs wide, the rotated pairs per axis hs / 4
    rows : uint
}

//! the 2D neox rope over q or k rows in place, the CPU `rope_neox_2d_rows`'s form: per head the x axis rotates the
//! pairs (j, j + pairs) of its first half against the row's x table, the y axis the same pairs of its second half
//! against the y table (one packed table row is [cx | sx | cy | sy], `pairs` each) - one thread per (row, head, pair), both axes
[ |> template_struct_instance]
class template GkRope2d {
    @ssbo @binding = 0 @role = "readwrite" v : array<float>
    @ssbo @binding = 1 @role = "read" tabs : array<float>
    @kargs pa : GkRope2dArgs

    def rope2d_body {
        let gid = gl_GlobalInvocationID.x
        let pairs = pa.hs / 4u
        let d = pa.nheads * pa.hs
        let per_row = max(1u, pa.nheads * pairs)
        if (gid < pa.rows * per_row) {
            let row = gid / per_row
            let hp = gid % per_row
            let base = row * d + (hp / max(1u, pairs)) * pa.hs
            let j = hp % max(1u, pairs)
            let tb = row * 4u * pairs
            let x0 = v[base + j]
            let x1 = v[base + j + pairs]
            let cx = tabs[tb + j]
            let sx = tabs[tb + pairs + j]
            v[base + j] = x0 * cx - x1 * sx
            v[base + j + pairs] = x0 * sx + x1 * cx
            let yb = base + pa.hs / 2u
            let y0 = v[yb + j]
            let y1 = v[yb + j + pairs]
            let cy = tabs[tb + 2u * pairs + j]
            let sy = tabs[tb + 3u * pairs + j]
            v[yb + j] = y0 * cy - y1 * sy
            v[yb + j + pairs] = y0 * sy + y1 * cy
        }
    }
}

struct GkWdecAttnArgs {
    dim : uint       //! a q row's width
    tstride : uint   //! the K/V planes' positions a head
    len0 : uint      //! the keys row 0 attends; a causal dispatch adds the row index
    causal : uint
    nheads : uint
    nrows : uint
    koff : uint      //! the layer's element base in kk / vv
    voff : uint
    qboff : uint     //! the q bias row's base in wn, folded at the load
    hs : uint        //! the head width where the stamp leaves HS at 0
    scale : float    //! the score scale, 1 / sqrt(hs)
}

//! the whisper decoder's chunked attention, the partials: one workgroup a (chunk, head, row) over 256 keys - the q bias folded
//! into the dot, the chunk's scores staged, their max and exp-sum landed as the (m, s) slot, the unnormalized weighted V as the
//! slot's row; the combine below merges the chunks by log-sum-exp
[ |> template_struct_instance]
class template GkWdecAttnPart : GkWgReduce {
    @template_constant HS : uint = 0u   //!< the head width served, 0 for `pa.hs`
    @ssbo @binding = 0 @role = "read" q : array<float>
    @ssbo @binding = 1 @role = "read" kk : array<float16>
    @ssbo @binding = 2 @role = "read" vv : array<float16>
    @ssbo @binding = 3 @role = "write" pms : array<float>    //! [nch][heads][rows][2]: the chunk's max and sum
    @ssbo @binding = 4 @role = "write" py : array<float>     //! [nch][heads][rows][hs]
    @ssbo @binding = 5 @role = "weight" wn : array<float>
    @kargs pa : GkWdecAttnArgs
    @workgroup scores : float[256]

    def wdec_head_width : uint {
        static_if (HS != 0u) {
            return HS
        } else {
            return pa.hs
        }
    }

    def wdec_attn_part_body {
        let hs = wdec_head_width()
        let nhr = max(1u, pa.nheads * pa.nrows)
        let nr = max(1u, pa.nrows)
        let c = gl_WorkGroupID.x / nhr
        let rem = gl_WorkGroupID.x % nhr
        let h = rem / nr
        let r = rem % nr
        let lid = gl_LocalInvocationID.x
        let len = pa.causal != 0u ? pa.len0 + r : pa.len0
        let tb = c * 256u
        let te = min(tb + 256u, len)
        let nloc = te > tb ? te - tb : 0u
        let slot = (c * pa.nheads + h) * pa.nrows + r
        let qbase = r * pa.dim + h * hs
        let kbase = pa.koff + h * pa.tstride * hs
        let vbase = pa.voff + h * pa.tstride * hs
        var t = lid
        while (t < nloc) {
            var s = 0.0
            var j = 0u
            while (j < hs) {
                s += (q[qbase + j] + wn[pa.qboff + h * hs + j]) * float(kk[kbase + (tb + t) * hs + j])
                s += (q[qbase + j + 1u] + wn[pa.qboff + h * hs + j + 1u]) * float(kk[kbase + (tb + t) * hs + j + 1u])
                s += (q[qbase + j + 2u] + wn[pa.qboff + h * hs + j + 2u]) * float(kk[kbase + (tb + t) * hs + j + 2u])
                s += (q[qbase + j + 3u] + wn[pa.qboff + h * hs + j + 3u]) * float(kk[kbase + (tb + t) * hs + j + 3u])
                j += 4u
            }
            scores[t] = s * pa.scale
            t += gl_WorkGroupSize.x
        }
        barrier()
        var mx = -3.0e38
        t = lid
        while (t < nloc) {
            mx = max(mx, scores[t])
            t += gl_WorkGroupSize.x
        }
        var m = 0.0
        wg_max_into(mx, 0u, m)
        var ss = 0.0
        t = lid
        while (t < nloc) {
            let e = exp(scores[t] - m)
            scores[t] = e
            ss += e
            t += gl_WorkGroupSize.x
        }
        var tot = 0.0
        wg_sum_into(ss, 1u, tot)   //! its barrier also lands every lane's exponentials before the output loop reads them across lanes
        if (lid == 0u) {
            pms[slot * 2u] = nloc > 0u ? m : -3.0e38
            pms[slot * 2u + 1u] = tot
        }
        var j = lid
        while (j < hs) {
            var acc = 0.0
            var tt = 0u
            while (tt < nloc) {
                acc += scores[tt] * float(vv[vbase + (tb + tt) * hs + j])
                tt++
            }
            py[slot * hs + j] = acc
            j += gl_WorkGroupSize.x
        }
    }
}

struct GkWdecCombArgs {
    dim : uint
    nchunks : uint
    nheads : uint
    nrows : uint
    hs : uint        //! the head width where the stamp leaves HS at 0
}

//! the chunked attention's combine over one row's head slice: the chunks' (m, s) slots merged by log-sum-exp, the lanes
//! striding the head's columns
[ |> template_struct_instance]
class template GkWdecAttnComb {
    @template_constant HS : uint = 0u   //!< the head width served, 0 for `pa.hs`
    @ssbo @binding = 0 @role = "read" pms : array<float>
    @ssbo @binding = 1 @role = "read" py : array<float>
    @ssbo @binding = 2 @role = "write" y : array<float>
    @kargs pa : GkWdecCombArgs

    def comb_head_width : uint {
        static_if (HS != 0u) {
            return HS
        } else {
            return pa.hs
        }
    }

    def wdec_attn_comb_body {
        let hs = comb_head_width()
        let nr = max(1u, pa.nrows)
        let h = gl_WorkGroupID.x / nr
        let r = gl_WorkGroupID.x % nr
        let lid = gl_LocalInvocationID.x
        var mg = -3.0e38
        var c = 0u
        while (c < pa.nchunks) {
            mg = max(mg, pms[((c * pa.nheads + h) * pa.nrows + r) * 2u])
            c++
        }
        var tot = 0.0
        c = 0u
        while (c < pa.nchunks) {
            let slot = (c * pa.nheads + h) * pa.nrows + r
            tot += pms[slot * 2u + 1u] * exp(pms[slot * 2u] - mg)
            c++
        }
        let inv = 1.0 / max(tot, 1.0e-30)
        var j = lid
        while (j < hs) {
            var acc = 0.0
            c = 0u
            while (c < pa.nchunks) {
                let slot = (c * pa.nheads + h) * pa.nrows + r
                acc += py[slot * hs + j] * exp(pms[slot * 2u] - mg)
                c++
            }
            y[r * pa.dim + h * hs + j] = acc * inv
            j += gl_WorkGroupSize.x
        }
    }
}

//! gemma4a's attention scales as the CPU chain rounds them: the query scale ahead of the per-dim scale and the key scale, both
//! into the log2-domain softmax
def gk_g4a_attn_scales(hs : int64) : tuple<qscale : float; kscale : float> {
    return (qscale = (1.0 / sqrt(float(hs))) / log(2.0), kscale = log(1.0 + exp(1.0)) / log(2.0))
}

struct GkRestrideArgs {
    rows : uint
    rows_pad : uint //! the padded panel's row cap (the scratch's, a multiple of the l column - `vt_cap_rows`): PAD writes zero rows past `rows` up to it, the unpad reads `rows`
    nheads : uint
    hs : uint
    hsp : uint
    sstride : uint  //! the compact rows' stride and the column their heads start at (a fused [q | k | v] row: 3d and 0 / d / 2d)
    soff : uint
    boff : uint     //! the BIAS stamps: the projection's bias row in wn, added as the compact row is read
    doff : uint     //! the padded panel's element base in its binding (the pad direction)
}

//! the head restride between the compact rows [rows x heads x hs] (at `sstride` / `soff`) and the attention tile's padded panel
//! [rows_pad x heads x hsp], one thread an element of the padded space: PAD writes the panel at `doff` with zero lanes and zero rows
//! past `rows` (f16 saturated under OUT16, the BIAS stamps adding the bias row as they read); the other way reads the compact rows back
[ |> template_struct_instance]
class template GkRestride {
    @template_constant PAD : bool = true
    @template_constant OUT16 : bool = false
    @template_constant OUT32 : bool = true   //! the f32 destination's gate: the f16 stamp clears it as it sets OUT16
    @template_constant BIAS : bool = false   //! the pad adds the bias row at `boff` (the whisper-class q / k / v biases fold into the restride)
    @ssbo @binding = 0 @role = "read" src : array<float>
    @ssbo @binding = 1 @role = "write" @template_gate = OUT32 dst : array<float>
    @ssbo @binding = 2 @role = "write" @template_gate = OUT16 dsth : array<float16>
    @ssbo @binding = 3 @role = "weight" @template_gate = BIAS wn : array<float>
    @kargs pa : GkRestrideArgs

    def restride_body {
        let gid = gl_GlobalInvocationID.x
        let per_row = pa.nheads * pa.hsp
        let rows_out = PAD ? pa.rows_pad : pa.rows
        if (gid < rows_out * per_row) {
            let row = gid / per_row
            let hp = gid % per_row
            let head = hp / pa.hsp
            let lane = hp % pa.hsp
            let ci = row * pa.sstride + pa.soff + head * pa.hs + lane
            static_if (PAD) {
                var v = 0.0
                if (lane < pa.hs && row < pa.rows) {
                    v = src[ci]
                    static_if (BIAS) {
                        v += wn[pa.boff + head * pa.hs + lane]
                    }
                }
                static_if (OUT16) {
                    dsth[pa.doff + gid] = half_sat(v)
                } else {
                    dst[pa.doff + gid] = v
                }
            } else {
                if (lane < pa.hs) {
                    static_if (OUT16) {
                        dsth[ci] = half_sat(src[gid])
                    } else {
                        dst[ci] = src[gid]
                    }
                }
            }
        }
    }
}


struct GkMelNormArgs {
    n_mel : uint
    n_frames : uint   //! the frames the mel holds
    valid : uint      //! the real frames: the statistics run over them, the frames past them read zero
    mfloor : float    //! the LOG_IN stamp's log guard
}

//! canary's per-feature normalization of a frame-major mel in place, one thread a mel column: the column's mean and unbiased
//! deviation over the valid frames, every valid frame normalized, the tail frames zero; the LOG_IN stamp takes the log of
//! the mel sums first (Metal's mel GEMM leaves the sums, Vulkan's mel kernel the logs)
[ |> template_struct_instance]
class template GkMelNorm {
    @template_constant LOG_IN : bool = false
    @ssbo @binding = 0 @role = "readwrite" mel : array<float>
    @kargs pa : GkMelNormArgs

    def mel_norm_body {
        let j = gl_GlobalInvocationID.x
        if (j < pa.n_mel) {
            let nm = pa.n_mel
            var sum = 0.0
            for (i in urange(pa.valid)) {
                static_if (LOG_IN) {
                    let v = log(mel[i * nm + j] + pa.mfloor)
                    mel[i * nm + j] = v
                    sum += v
                } else {
                    sum += mel[i * nm + j]
                }
            }
            let mean = sum / float(max(1u, pa.valid))
            var sq = 0.0
            for (i in urange(pa.valid)) {
                let dv = mel[i * nm + j] - mean
                sq += dv * dv
            }
            let denom = sqrt(sq / max(float(pa.valid) - 1.0, 1.0)) + 1e-5
            for (i in urange(pa.valid)) {
                mel[i * nm + j] = (mel[i * nm + j] - mean) / denom
            }
            for (i in urange(pa.valid, pa.n_frames)) {
                mel[i * nm + j] = 0.0
            }
        }
    }
}

struct GkQ3aShuffleArgs {
    nch : uint          //! the conv channel count
    ncp : uint          //! a conv3 row's stride (the channels, or Metal's 64-padded ones)
    chunk_stride : uint //! a chunk's elements in y3 (16 x 13 rows of ncp, or Metal's row-padded chunk)
    npos : uint         //! the window's positions
}

//! the qwen3a feature rows off the third conv stage, the conv_out GEMM's feed: feat[ci][ow][h + 16 c] = y3[ci][(h x 13 + ow) x ncp + c],
//! thirteen positions a chunk; the stamp names the feed's element type `OT` (the f16 tile's halves, the f32 tile's floats)
[ |> template_struct_instance]
class template GkQ3aShuffle {
    @ssbo @binding = 0 @role = "read" y3 : array<float>
    @ssbo @binding = 1 @role = "write" feat : array<OT>
    @kargs pa : GkQ3aShuffleArgs

    def q3a_shuffle_body {
        let gid = gl_GlobalInvocationID.x
        let feat_w = max(1u, 16u * pa.nch)
        if (gid < pa.npos * feat_w) {
            let tok = 13u
            let row = gid / feat_w
            let j = gid % feat_w
            let h = j % 16u
            let c = j / 16u
            let ci = row / tok
            let ow = row % tok
            feat[gid] = OT(y3[ci * pa.chunk_stride + (h * tok + ow) * pa.ncp + c])   // nolint:PERF020,LINT019 — the f16 stamp narrows here; on the f32 stamp the cast is the identity
        }
    }
}

struct GkIm2col2dArgs {
    iw : uint         //! the source block's width (frames), height (mel rows / channels' rows) and channel count
    ih : uint
    cin : uint
    sstride : uint    //! a source pixel's stride: cin, or the padded channel rows a GEMM wrote
    kdp : uint        //! the column row's padded width, a tile multiple; the columns past cin x 9 read zero
    rows : uint       //! the output rows of every chunk: chunks x oh x ow
    soff : uint       //! the source's element base in its binding
}

//! the im2col of a k3 s2 p1 conv2d over a window's chunks, the conv GEMM's feed: row r = (chunk, oy, ox), column k = (ic x 3 + kh)
//! x 3 + kw - the file's weight order, so the weights need no permute; the source rows are [chunk][ih][iw][sstride]; the stamp
//! names the feed's element type `OT`
[ |> template_struct_instance]
class template GkIm2col2d {
    @ssbo @binding = 0 @role = "read" src : array<float>
    @ssbo @binding = 1 @role = "write" col : array<OT>
    @kargs pa : GkIm2col2dArgs

    def im2col2d_body {
        let gid = gl_GlobalInvocationID.x
        let kdp = max(1u, pa.kdp)
        if (gid < pa.rows * kdp) {
            let r = gid / kdp
            let k = gid % kdp
            var v = 0.0
            if (k < pa.cin * 9u) {
                let ow = (pa.iw - 1u) / 2u + 1u
                let oh = (pa.ih - 1u) / 2u + 1u
                let per_chunk = max(1u, oh * ow)
                let ci = r / per_chunk
                let rr = r % per_chunk
                let ic = k / 9u
                let kh = (k % 9u) / 3u
                let kw = k % 3u
                let iy = int((rr / ow) * 2u + kh) - 1
                let ix = int((rr % ow) * 2u + kw) - 1
                if (iy >= 0 && iy < int(pa.ih) && ix >= 0 && ix < int(pa.iw)) {
                    v = src[pa.soff + ci * (pa.ih * pa.iw * pa.sstride) + (uint(iy) * pa.iw + uint(ix)) * pa.sstride + ic]
                }
            }
            col[gid] = OT(v)   // nolint:PERF020,LINT019 — the f16 stamp narrows here; on the f32 stamp the cast is the identity
        }
    }
}
