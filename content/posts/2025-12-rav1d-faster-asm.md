---
title: "Why is calling my asm function from Rust slower than calling it from C?"
summary: ""
date: 2025-12-27T10:00:00+00:00
tags: ["rust", "c", "performance"]
type: post
showTableOfContents: true
image: "/2025-12-rav1d-faster-asm/asm_side_by_side.webp"
weight: 1
---

This is a follow-up to [making the rav1d video decoder 1% faster]({{< ref "/posts/2025-05-rav1d-faster.md" >}}),
where we compared profiler snapshots of `rav1d` (the Rust implementation) and `dav1d` (the C baseline)
to find specific functions that were slower in the Rust implementation[^1].

Today, we are going to pay off a small debt from that post: since `dav1d` and `rav1d` share the same hand-written assembly functions,
we used them as **anchors** to navigate the different implementations - _they_, at least, should match exactly!
And they did. Well, _almost_ all of them did.

This, dear reader, is the story of the one function that _didn't_. 

[^1]: Given a specific function that is known to be slow, one can usually compare the implementations and find the culprit - we found an unneeded zero-initialization of a large buffer and missing optimized equality comparisons for a number of small structs. We actually _also_ saw that one of these structs, `Mv`, was not 4-byte aligned, which I was too quick to dismiss - [@daxtens](https://github.com/daxtens) got an additional %1 improvement by fixing this in [this PR](https://github.com/memorysafety/rav1d/pull/1433).

## An Overview

We’ll need to ask - and answer! - [three 'Whys'](https://en.wikipedia.org/wiki/Five_whys) today: \
Using the same [techniques] from last time,
we'll see that a specific assembly function is, indeed, slower in the Rust version.

[techniques]: {{< ref "/posts/2025-05-rav1d-faster.md" >}}#background-and-approach

1. But why? ➡️ Because **loading data** in the Rust version is slower, which we discover using `samply`'s special asm view. [1](#looking-at-the-opcodes)
2. But why? ➡️ Because the Rust version stores much **more data on the stack**, which we find by playing with some arguments and looking at the generated LLVM IR. [2](#a-good-guess)
3. But why? ➡️ Because **the compiler cannot optimize** away a specific Rust abstraction across function pointers! [3](#from-top-to-bottom)

Which we fix by switching to a more compiler-friendly version ([PR]). [4](#switch-it-up)

[PR]: https://github.com/memorysafety/rav1d/pull/1418

<i>Side note: again, we'll be running all these benchmarks on a MacBook, so our tools are a tad limited
and we'll have to resort to some guesswork. Leave a comment if you know more - or, even better, write an article about profiling on macOS 🍎💨.</i>

Discuss on [r/rust](https://www.reddit.com/r/rust/comments/1pwzti4/why_is_calling_my_asm_function_from_rust_slower/), [lobsters](https://lobste.rs/s/byxxmk/why_is_calling_my_asm_function_from_rust), [HN](https://news.ycombinator.com/item?id=46401982)! 👋

<p align="center">
    <img src="/2025-12-rav1d-faster-asm/instruments.webp" 
        alt="Instruments quit unexpectedly after running a few recordings" loading="lazy" width="66%" width="1458px" height="622px" />
</p>

## `filter4_pri_edged_8bpc`

Let's rerun the benchmark after the previous post's changes:

```bash
./rav1d $ git checkout cfd3f59 && cargo build --release
./rav1d $ sudo samply record ./target/release/dav1d -q -i Chimera-AV1-8bit-1920x1080-6736kbps.ivf -o /dev/null --threads 1
```

We'll switch to the inverted call stack view and filter for the `cdef_` functions, resulting in the following clippings[^2].
The assembly functions are the ones with the `_neon` suffix.

[^2]: Again, these are non-interactive clippings, created using the excellent [Save Page WE](https://chromewebstore.google.com/detail/save-page-we/dhhpefjklgkmgeafimnjhojgjamoafof) extension and creative use of _Delete element_. See more in the previous. post's [profiling section]({{< ref "/posts/2025-05-rav1d-faster.md" >}}#profiling).

<style>
/* Use a responsive grid to show the clippings side by side if the screen is wide enough */
.wrapper {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-gap: 2px;
}

.box {
  padding: 2px;
}

.asm-intro-top-bottom {
  display: none;
}

@media (max-width: 520px) {
  .wrapper {
    grid-template-columns: 1fr; /* two rows */
  }

  .asm-intro-left-right {
    display: none;
  }

  .asm-intro-top-bottom {
    display: block;
  }
}
</style>

<p class="asm-intro-left-right">
{{< markdownify >}}
On the left is `dav1d` (C), and on the right `rav1d` (Rust):
{{< /markdownify >}}
</p>

<p class="asm-intro-top-bottom">
{{< markdownify >}}
On the top is `dav1d` (C), and on the bottom `rav1d` (Rust):
{{< /markdownify >}}
</p>

<div class="wrapper">
    <div class="box">
        <iframe src="/2025-12-rav1d-faster-asm/dav1d_cdef_inverted.html" loading="lazy" width="100%" height="300"></iframe>
    </div>
    <div class="box">
        <iframe src="/2025-12-rav1d-faster-asm/rav1d_cdef_inverted.html" loading="lazy" width="100%" height="300"></iframe>
    </div>
</div>

Looking at the sample count, most of the functions match (to within ~10%)[^4], except the highlighted `cdef_filter4_pri_edged_8bpc_neon` which is **30% slower**. 
We see a difference of 350 samples. Sampling at 1000 Hz, this corresponds to 0.35 seconds, or ~0.5% of the total runtime.

[^4]: `dav1d_cdef_padding4_edged_8bpc_neon` is also a bit slower - but it's a smaller function overall, so we're going to ignore that.

This is _very sus_: obviously this is the exact same function, and barring a logical bug in the implementation,
it must process the exact same data.

So how can this be?

## Looking at the Opcodes

Luckily for us, `samply` has _exactly_ what we need here: we can get into the `asm` view by double-clicking on the function,
which shows **a per-instruction sample count**.

And it seems that fortune favors the bold, 
because we find the entire difference in _a single instruction_ less than 25 lines into the call.

Let's look at the `ld1 {v0.s}[2], [x13]` line, highlighted below in yellow. \
It appears in 10 samples in the `dav1d` run (C), but in 441 (!) samples in the `rav1d` run (Rust):

<div class="wrapper">
    <div class="box">
        <iframe src="/2025-12-rav1d-faster-asm/dav1d_cdef_asm.html" loading="lazy" width="100%" height="460"></iframe>
    </div>
    <div class="box">
        <iframe src="/2025-12-rav1d-faster-asm/rav1d_cfd3f59_cdef_asm.html" loading="lazy" width="100%" height="460"></iframe>
    </div>
</div>

At this point, you might be wondering: what is `ld1`? What's that `{v0.s}[2]` syntax? 
And... why is `x13` _that_ different from `x2`, `x12`, or `x14`?

### `ld1`

Let's try to decode what `ld1 {v0.s}[2], [x13]` means.

A quick search leads us to the [LD1 page in the Arm A-profile A64 Instruction Set Architecture documentation](https://developer.arm.com/documentation/ddi0602/2025-09/SIMD-FP-Instructions/LD1--single-structure---Load-one-single-element-structure-to-one-lane-of-one-register-?lang=en), which helpfully says the following:

> **LD1** - Load one single-element structure to one lane of one register
> 
> This instruction loads a single-element structure from memory and writes the result to the specified lane of the SIMD&FP register

It also explains that `v0` is a SIMD register, and `.s` is the 32-bit variant of this instruction.

So, TL;DR: this instruction loads data from the address in the `x13` register into lane 2 of the `v0` SIMD register.

Which means that the three adjacent instructions _also_ do almost the exact same thing.

## A Good Guess

Ignoring the start of the function, let's look at the lines that appear right before the load instructions:


<style>
/* Highlighted code samples have a black background for some reason  */
.rust-highlighted-code > pre > code {
    background-color: unset;
}
</style>

```asm {hl_lines=[6] class="rust-highlighted-code"}
add x12, x2, #0x8
add x13, x2, #0x10
add x14, x2, #0x18
ld1 {v0.s}[0], [x2]  ; Fast - 20 samples.
ld1 {v0.s}[1], [x12] ; Fast - 16 samples.
ld1 {v0.s}[2], [x13] ; Slow - 441 samples.
ld1 {v0.s}[3], [x14] ; Fast as well!
```

_PSA: if you don't see syntax highlighting, [disable the 1Password extension](https://www.1password.community/discussions/developers/1password-chrome-extension-is-incorrectly-manipulating--blocks/165639)._

Seems simple enough - we load 32-bit values from the addresses at `x2 + {0,8,16,24}` into `v0`. But what address is stored in `x2`? \
On AArch64, integer and pointer **parameters** are passed in `x0` through `x7`, and sure enough,
looking at the `extern "C" fn` definition, we find:

```rust
unsafe extern "C" fn filter(
    dst: *mut DynPixel,           // x0
    dst_stride: ptrdiff_t,        // x1
    tmp: *const MaybeUninit<u16>, // x2
    // ...
) -> ()
```

Our old friend `tmp`! We [saw] in the previous post that these assembly functions are dispatched from a function called `cdef_filter_neon_erased`. 
This function defines `tmp` on the stack as a buffer of (uninitialized) `u16`s, and _partially_ fills it using a padding function which is also written in assembly.

So, why would reading from a contiguous smallish buffer be slow for one particular part of that buffer?

**At this point, we are going to take a guess** (leave a comment if you know more!): 
there's likely a caching issue somewhere that causes the CPU to stall for that particular load.

But why? Maybe it's something in the way data is _written_ to the buffer?
Time to take a closer look. In particular, there's something a bit unexpected in the _arguments_ of the `cdef_filter_neon_erased` function:

[saw]: {{< ref "/posts/2025-05-rav1d-faster.md" >}}#cdef_filter_neon_erased

```rust
unsafe extern "C" fn cdef_filter_neon_erased<BD: BitDepth, ..>(
    dst: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    ..,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    _top: *const FFISafe<CdefTop>,
    _bottom: *const FFISafe<CdefBottom>,
) {
    let mut tmp_buf = Align16([MaybeUninit::uninit(); TMP_LEN]);
    let tmp = &mut tmp_buf.0[..];

    padding::Fn::neon::<BD, W>().call::<BD>(  //
        tmp,                                  // <--- Fills tmp by calling a `cdef_padding_XYZ_neon` function. 
        dst, stride, left, top, bottom, ..    // 
    );
    filter::Fn::neon::<BD, W>().call(         // <--- Calls the specific `cdef_filter_XYZ_neon` function.
        dst,
        stride,
        tmp,
        ..
    )
}
```

This is... a bit much, but as you can imagine, `dav1d` _doesn't_ have the last 3 arguments (an `_` in Rust denotes an unused variable). 
Looking around some more, they are only used in a function called <code>cdef_filter_<strong>block_c</strong>_erased</code>, which is - despite the name - a pure-Rust fallback in case the asm functions are unavailable.

I wonder what will happen if we... if we just remove them?

<picture>
  <source
    media="(max-width: 820px)"
    srcset="/2025-12-rav1d-faster-asm/why_shouldnt_i.webp"
    width="484"
    height="484"
    loading="lazy"
  >
  <img
    src="/2025-12-rav1d-faster-asm/why_shouldnt_i_lg.webp"
    width="968"
    height="242"
    loading="lazy"
    alt="Bilbo - Why shouldn’t I keep it? Meme"
  >
</picture>


### A "Fix"

If we do remove them:

```diff
-    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
-    _top: *const FFISafe<CdefTop>,
-    _bottom: *const FFISafe<CdefBottom>,
+    // _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
+    // _top: *const FFISafe<CdefTop>,
+    // _bottom: *const FFISafe<CdefBottom>,
```

and (temporarily) replace <code>cdef_filter_<strong>block_c</strong>_erased</code> with a stub:

```rust
unsafe extern "C" fn cdef_filter_block_c_erased<BD: BitDepth, const W: usize, const H: usize>(
    _dst_ptr: *mut DynPixel,
    ...
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    // dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    // top: *const FFISafe<CdefTop>,
    // bottom: *const FFISafe<CdefBottom>,
) {
    todo!()
}
```

When we re-run our benchmark, we see something cool:

Our dear `cdef_filter4_pri_edged_8bpc_neon`, which accounted for 1,562 samples before, is now **down to 1,268 samples** (now within 5% of  `dav1d`'s 1,199),
and all our `ld1` (memory load) instructions are down to `dav1d` levels! No more stalling.

<iframe src="/2025-12-rav1d-faster-asm/rav1d_without_args_asm_min.html" loading="lazy" width="100%" height="135px"></iframe>

Huzzah! Or... Huzzah?

<p align="center">
    <img src="/2025-12-rav1d-faster-asm/guy_stopping_friend_from_a_girl_meme_small.webp" 
        alt="Guy putting hand on other guy chasing a girl meme" loading="lazy" width="50%" width="627px" height="472px" />
</p>

## An Elegant Weapon for a More Civilized Age

Let's recap: On the one hand, we found a meaningful slowdown between the Rust and the C versions,
and we even managed to create a "fixed" version that doesn't exhibit the same problem.

On the other hand, we only have _vibes_ about what the problem is (memory is haunted?), nothing about the fix makes sense (removing unused stuff helps how?), _and_ we can't use this code because we removed an important fallback.

The only silver lining is that because we have a faster version, we can try to compare it to the original and find out what changed,
and that might lead us to the real fix.

Which is where `cargo asm` comes into play.

Our theory is that _something_ is different in the way the memory is laid out between the versions. \
We'll guess that it's probably something with the stack, because (a) removing arguments made a difference, and arguments are (sometimes) passed on the stack and (b) all the heap data structures closely follow the original `dav1d` ones, and there aren't that many of them anyway.

So what can `cargo asm` tell us?

### Peeking Under the Hood

We can compare `cdef_filter_neon_erased`, using either `--asm` or `--llvm` modes[^3], but long story short, there doesn't seem to be any differences between the baseline and the faster version. Which at least makes sense - we didn't change anything about this function because it wasn't using those arguments in the first place!

[^3]: Using `cargo asm -p rav1d --lib --asm cdef_filter_neon_erased 2` and `cargo asm -p rav1d --lib --llvm cdef_filter_neon_erased 2`.

But what if we go one level up? `_erased` is called from a function named `rav1d_cdef_brow` (which we also [briefly saw] in the last post),
which is a very complex, 300-line behemoth. 
However, it seems like this function receives its data via a few nice structs, which means that either one of them is messed up - which is relatively easy to check - or that the problem is somewhere **inside** this function.

[briefly saw]: {{< ref "/posts/2025-05-rav1d-faster.md" >}}##avoid-needlessly-zeroing-buffers-with-maybeuninit


```rust
fn rav1d_cdef_brow<BD: BitDepth>(
    c: &Rav1dContext,
    tc: &mut Rav1dTaskContext,
    f: &Rav1dFrameData,
    p: [Rav1dPictureDataComponentOffset; 3],
    // .. a few simple arguments ..
) { ... }
```

And this time, `cargo asm`[^7] lights up like a Christmas tree 🎄. \
Here's our faster version:

[^7]: Specifically, `cargo asm -p rav1d --lib --llvm rav1d_cdef_brow 1`. We can also verify that the stack allocations are part of the final binary by running with `--asm` instead.

```llvm
; rav1d::cdef_apply::rav1d_cdef_brow
; Function Attrs: nounwind
define internal fastcc void @rav1d::cdef_apply::rav1d_cdef_brow(...) {
start:
  %dst.i = alloca [16 x i8], align 8
  %variance = alloca [4 x i8], align 4
  %lr_bak = alloca [96 x i8], align 16
  %_17 = icmp sgt i32 %by_start, 0
  %. = select i1 %_17, i32 12, i32 8
  ...
}
```

And here's the baseline version:

```llvm
; rav1d::cdef_apply::rav1d_cdef_brow
; Function Attrs: nounwind
define internal fastcc void @rav1d::cdef_apply::rav1d_cdef_brow(...) {
start:
  %top.i400 = alloca [16 x i8], align 8
  %dst.i401 = alloca [16 x i8], align 8
  %top.i329 = alloca [16 x i8], align 8
  %dst.i330 = alloca [16 x i8], align 8
  %top.i = alloca [16 x i8], align 8
  %dst.i317 = alloca [16 x i8], align 8
  %dst.i = alloca [16 x i8], align 8
  %bot5 = alloca [24 x i8], align 8
  %bot = alloca [24 x i8], align 8
  %variance = alloca [4 x i8], align 4
  %lr_bak = alloca [96 x i8], align 16
  %_17 = icmp sgt i32 %by_start, 0
  %. = select i1 %_17, i32 12, i32 8
  ...
}
```


Which means that somehow, the baseline version allocates on the stack - using `alloca` - 144 bytes more than the faster version!
It would also seem that all these extra allocations are for multiple instances of `dst`, `top`, and `bot` (i.e., `bottom`), 
which matches the arguments we removed in the faster version.

So now we only need to... not do that, I guess?

## From Top to Bottom

Our revised but incomplete theory is thus:

(1) `cdef_filter4_pri_edged_8bpc_neon` reads data from or via `dst`, `top` and/or `bot`, which ends up affecting the third `ld1` line.

{{< detail-tag "More" >}}

The calls to the `filter` functions are defined like this:

```rust
unsafe extern "C" fn filter(
    dst: *mut DynPixel,
    dst_stride: ptrdiff_t,
    tmp: *const MaybeUninit<u16>,
    // ..
) -> { .. }
```

and the assembly function _template_ is located in `src/arm/64/cdef.S`:
```asm
// void cdef_filterX_edged_8bpc_neon(pixel *dst, ptrdiff_t dst_stride,
//                                   const uint8_t *tmp, int pri_strength,
//                                   int sec_strength, int dir, int damping,
//                                   int h);
.macro filter_func_8 w, pri, sec, min, suffix
function cdef_filter\w\suffix\()_edged_8bpc_neon
    // ..
    ld1             {v0.s}[0], [x2]             // px
    ld1             {v0.s}[1], [x12]            // px
    ld1             {v0.s}[2], [x13]            // px
    ld1             {v0.s}[3], [x14]            // px
```
{{< /detail-tag >}}

(2) `cdef_filter_neon_erased` accepts **two** sets of these, one as raw pointers for the asm version and one as these `*FFISafe` pointers that are only used in the pure-Rust version.

{{< detail-tag "More" >}}

The assembly dispatch function (`_erased`) only uses the `*mut DynPixel` versions:

```rust
pub unsafe extern "C" fn cdef_filter_neon_erased<BD: BitDepth, const W: usize, const H: usize, .. >(
    dst: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    // ..
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    _top: *const FFISafe<CdefTop>,
    _bottom: *const FFISafe<CdefBottom>,
) {
    // ...
    padding::Fn::neon::<BD, W>().call::<BD>(tmp, dst, stride, left, top, bottom, H, edges);
    filter::Fn::neon::<BD, W>().call(dst, stride, tmp, pri_strength, sec_strength, dir, damping, H, edges, bd);
}
```

While the pure Rust version uses only the fully typed and safe `[BD::Pixel]` versions:

```rust
fn cdef_filter_block_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    dst: Rav1dPictureDataComponentOffset,
    left: &[LeftPixelRow2px<BD::Pixel>; 8],
    top: CdefTop,
    bottom: CdefBottom,
    // ...
) { .. }
```

{{< /detail-tag >}}

(3) `rav1d_cdef_brow` sets up all of these in a few different ways, probably for the different variations of `cdef_filter4_{pri_edged,pri_sec_edge,sec_edge,sec,pri}_8bpc_neon`.

{{< detail-tag "More" >}}

For example, this is a small unedited part of `rav1d_cdef_brow`. See how `top` and `bot` have a non-trivial setup:
```rust
let (top, bot) = top_bot.unwrap_or_else(|| {
    let top = WithOffset {
        data: &f.lf.cdef_line_buf,
        offset: f.lf.cdef_line[tf as usize][0],
    } + have_tt as isize * (sby * 4) as isize * y_stride
        + (bx * 4) as isize;
    let bottom = bptrs[0] + (8 * y_stride);
    (top, WithOffset::pic(bottom))
});

if y_pri_lvl != 0 {
    let adj_y_pri_lvl = adjust_strength(y_pri_lvl, variance);
    if adj_y_pri_lvl != 0 || y_sec_lvl != 0 {
        f.dsp.cdef.fb[0].call::<BD>(
            bptrs[0],
            &lr_bak[bit as usize][0],
            top,
            bot,
            adj_y_pri_lvl,
            y_sec_lvl,
            dir,
            damping,
            edges,
            bd,
        );
    }
}
```

{{< /detail-tag >}}

Having the two sets of pointers prevents the compiler from performing some optimizations,
and it just so happens that this results in a layout that causes the CPU to stall.

There's _so much more_ going on here, but let's keep our focus and try to actually fix the issue at hand.

### Why is it `FFISafe`-ed?

Simplified, `rav1d_cdef_brow` sets up `top` like so:

```rust
let cdef_line_buf: AlignedVec64<u8>;

let top = WithOffset {
    data: &cdef_line_buf,
    offset,
} + ... as isize;
```

with `dst` and `bottom` following similar patterns. \
Checking `WithOffset`, we see that it's a utility for accessing a buffer using an index:

```rust
#[derive(Clone, Copy)]
pub struct WithOffset<T> {
    pub data: T,
    pub offset: usize,
}

impl<T> AddAssign<usize> for WithOffset<T> { .. }
impl<T> SubAssign<usize> for WithOffset<T> { .. }
// A few more impl like this.

impl<P: Pixels> WithOffset<P> {
    pub fn as_ptr<BD: BitDepth>(&self) -> *const BD::Pixel {
        self.data.as_ptr_at::<BD>(self.offset)
    }
 
    // A few more of these as well.
}
```

Looking at this struct, we start to see what's going on: `WithOffset` is, on a 64-bit architecture, the size of `T` plus 8 bytes, which matches the `alloca` calls of 16 and 24 bytes we saw before.

It is also not "FFI-safe", which means that passing it as an argument in an `extern "C"` function - such as our asm functions - is [somewhat controversial](https://github.com/rust-lang/rust/issues/116963),
and `rav1d` gets around that by having this special `FFISafe` struct that makes this problem magically[^5] go away.

Because `WithOffset` is a buffer-access utility, it can be used to create raw pointers into the underlying buffer.
But because the safe Rust fallback doesn't want raw pointers, we end up having both versions when we [`call`] 
either the asm or the Rust version of the function:

```rust
let top_ptr: *mut DynPixel = top.as_ptr::<BD>().cast();
let bottom_ptr: *mut DynPixel = bottom.wrapping_as_ptr::<BD>().cast();
let top = FFISafe::new(&top);
let bottom = FFISafe::new(&bottom);

// We're simplifying here, and we also ignore the differences between u8, DynPixel and BD::Pixel.
pub type CdefTop<'a> = WithOffset<&'a u8>;

// A function pointer to the best available impl...
let callback: extern "C" fn(
    .., 
    top_ptr: *mut DynPixel, 
    .., 
    top: *const FFISafe<CdefTop>,
) = /* ... selected at runtime */;

// Maybe end up in Rust, maybe in assembly, who knows!
callback(.., top_ptr, bottom_ptr, .., top, bottom);
```

[`call`]: https://github.com/memorysafety/rav1d/blob/25e5574/src/cdef.rs#L70
[^5]: See [ffi_safe.rs](https://github.com/memorysafety/rav1d/blob/25e5574/src/ffi_safe.rs#L11)

OK! Phew! Wow! This is great (or, sorry that happened to you), but what can we do about this?

### Switch It Up

> Move it up, down, left, right, oh - Switch it up like Nintendo ~ S. A. Carpenter

Because we have this ` *const FFISafe<WithOffset<..>>` at an `extern "C"` function boundary, 
the compiler is more limited in what it can do with the values of `top`, `bottom`, and `dst`.

What if we switched it up? \
We can make `WithOffset` FFI-safe by slapping a `#[repr(C)]` on it, as long as `T` is FFI-safe:

```rust
#[derive(Clone, Copy)]
#[repr(C)] // <- New!
pub struct WithOffset<T> {
    pub data: T,
    pub offset: usize,
}
```

Then, we can change each variable from `*const FFISafe<WithOffset<?>>` to \
`WithOffset<*const FFISafe<?>>`.

For example, before we had something like:
```rust
top: *const FFISafe<WithOffset<&'a u8>>
```

We can change that to:
```rust
top: WithOffset<*const FFISafe<&'a u8>>
```

The key difference is that now, instead of creating an FFI-safe pointer to our arguments,
we actually destructure them and create new instances of `WithOffset`:

```rust
let top: WithOffset<&'a u8> = /* an argument */;

// Used to be `let top = FFISafe::new(&top)`.
let top = WithOffset {
    data: FFISafe::new(&top.data),
    offset: top.offset,
};
```

This should - in theory - let the compiler see we only use a single instance of each parameter at any given time.

But does it?

## Will It Blend?

We can use the same shtick for `dst` and `bot`,
and the final diff turns out shorter than this article 🫠.

{{< detail-tag "Click to see the full diff" >}}
  ```patch{{% include "2025-12-rav1d-faster-asm-full-diff.patch" %}}
  ```
{{< /detail-tag >}}

Now we can run `cargo asm` again:

```llvm
; rav1d::cdef_apply::rav1d_cdef_brow
; Function Attrs: nounwind
define internal fastcc void @rav1d::cdef_apply::rav1d_cdef_brow(...) {
start:
  %dst.i = alloca [16 x i8], align 8
  %bot5 = alloca [24 x i8], align 8
  %bot = alloca [24 x i8], align 8
  %variance = alloca [4 x i8], align 4
  %lr_bak = alloca [96 x i8], align 16
  %_17 = icmp sgt i32 %by_start, 0
  %. = select i1 %_17, i32 12, i32 8
```

It's not _perfect_ - we didn't have these extra `bot` and `bot5` in our [original fix] - but it's much better! Let's run the profiler 🎶 _one last time_ 🎶.

[original fix]: #peeking-under-the-hood

Remember: `cdef_filter4_pri_edged_8bpc_neon` had 1,562 samples in the slow Rust baseline, vs. 1,199 in `dav1d`.

<iframe src="/2025-12-rav1d-faster-asm/rav1d_patched_cdef_asm.html" loading="lazy" width="100%" height="400px"></iframe>

Yes! We are down from 1,562 **to 1,260!** samples (which is within 5% of `dav1d`), \
_and_ the `ld1` lines are no longer slow, _and_ the pure-Rust fallback works as expected.

Huzzah!
