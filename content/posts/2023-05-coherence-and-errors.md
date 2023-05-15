---
title: "Winning the fight against the Rust compiler (Coherence in Rust, feat. rustc sources)"
date: 2023-05-10T12:00:00+00:00
tags: ["rust"]
type: post
showTableOfContents: true
image: "/2023-05-coherence-and-errors/orphan-checker-preview.png"
---

## An unexpected error

A friend was experimenting with Rust and asked if I could help decipher an error message for them:

```ps
PS Z:\Projects\aoc\src\bin> cargo build --bin=2021_day19
   Compiling aoc v0.1.0 (Z:\Projects\aoc)
error[E0210]: type parameter `T` must be covered by another type when it appears before the first local type (`VecN<T, N>`)
  --> src\vec.rs:65:6
   |
65 | impl<T: MulAssign, const N: usize> Mul<VecN<T, N>> for T {
   |      ^ type parameter `T` must be covered by another type when it appears before the first local type (`VecN<T, N>`)  
   |
   = note: implementing a foreign trait is only possible if at least one of the types for which it is implemented is local, and no uncovered type parameters appear before that first local type
   = note: in this case, 'before' refers to the following order: `impl<..> ForeignTrait<T1, ..., Tn> for T0`, where `T0` is the first and `Tn` is the last
```

And, I could! I actually updated this error message (in [#66253](https://github.com/rust-lang/rust/pull/66253)) after helping stabilize [RFC 2451 (re-rebalance coherence)](https://github.com/rust-lang/rust/issues/63599) in [#65879](https://github.com/rust-lang/rust/pull/65879).

But this error is indeed strange.

Isn’t one of the reasons for using traits is to allow others to implement them? 
Why does Rust reject this particular combination of structs, traits and generics? 

And.. can we modify `rustc` and force it to accept our code?

Read on to learn about _coherence_, the orphan rules and why they actually *increase* the usability of traits, 
and find out what happens when you fight the compiler and win (only to lose the war).

**Table of Contents**

1. [A minimal example](#a-minimal-example)
2. [Why](#why)
3. [Cheat sheet](#cheat-sheet)
4. [How (exploring `rustc` sources)](#how)
5. [Summary](#summary)
6. [Outro](#outro)

## A minimal example

Let's start with something that does compile.

We'll define our own scalar wrapper and make it generic over the actual numerical type (`u32`/`f32`/etc).

```rust
struct Scalar<T>(T);
```

A struct with one unamed field, nothing too fancy.

This being a scalar, let's implement multiplication:

```rust
/// ex0.rs
use std::ops::Mul;

struct Scalar<T>(T);

impl<T> Mul<T> for Scalar<T>
where
    T: Mul<Output = T>,
{
    type Output = T; // Means that, for example, Scalar<u32> * u32 = u32.
    fn mul(self, s: T) -> T { self.0 * s }
}

fn main() {
    let s = Scalar(7u32);
    let n = 6u32;
    println!("{}", s * n);
}
```

```bash
$ rustc ex0.rs
$ ./ex0
42
```

Boom.

But... what about multiplication on the right? Luckily we didn't choose to implement a matrix, so this should be easy!

```diff
-println!("{}", s * n);
+println!("{}", n * s);
```

```bash
$ rustc ex0.rs
error[E0277]: cannot multiply `u32` by `Scalar<u32>`
  --> ex0.rs:17:22
   |
17 |     println!("{}", n * s);
   |                      ^ no implementation for `u32 * Scalar<u32>`
   |
   = help: the trait `Mul<Scalar<u32>>` is not implemented for `u32`
```

`rustc` helpfully points us in the right direction: for `T == u32`, we have `impl<T> Mul<T> for Scalar<T>`, but not `impl<T> Mul<Scalar<T>> for T`.

We could implement this as `fn mul(self, s: Scalar<T>) -> T { self * s.0 }`, so it shouldn't be a problem, right?

```rust
/// ex1.rs
use std::ops::Mul;

struct Scalar<T>(T);

// For s * n.
impl<T: Mul<Output = T>> Mul<T> for Scalar<T> {}

// For n * s.
impl<T: Mul<Output = T>> Mul<Scalar<T>> for T {}

fn main() {}
```

Wrong.

```bash
$ rustc ex1.rs
error[E0210]: type parameter `T` must be covered by another type when it appears before the first local type (`Scalar<T>`)
 --> ex1.rs:9:6
  |
9 | impl<T: Mul<Output = T>> Mul<Scalar<T>> for T {}
  |      ^ type parameter `T` must be covered by another type when it appears before the first local type (`Scalar<T>`)
  |
  = note: implementing a foreign trait is only possible if at least one of the types for which it is implemented is local, and no uncovered type parameters appear before that first local type
  = note: in this case, 'before' refers to the following order: `impl<..> ForeignTrait<T1, ..., Tn> for T0`, where `T0` is the first and `Tn` is the last
```

And now that we have a concrete non-compiling example. Let's dive in!

## Why

The reason this example fails to compile if because of _coherence_.

While the term _coherence_ ("the quality of being logical and consistent") is used in a few different ways in math and computer science ([wikipedia](https://en.wikipedia.org/wiki/Coherence)), in the realm of type systems, [RFC 2451](https://github.com/rust-lang/rfcs/pull/2451) gives the following useful definition:

> Coherence means that for any given trait and type, there is one specific implementation that applies.

This can also explained with an example.

Rust permits:

```rust
/// bignum_lib.rs
use std::ops::Mul;

pub struct BigNum { }

// Implements BigNum * T.
impl<T> Mul<T> for BigNum {
    type Output = BigNum;
    fn mul(self, _: T) -> Self::Output { todo!() }
}
```

So it has to reject:

```rust
// Implements T * Scalar<T>
impl<T> Mul<Scalar<T>> for T {}
```

Why? If both are allowed and we use the `BigNum` library, the compiler will have two competing implementations to call when someone uses `BigNum * Scalar<BigNum>`.

Unless there are specific rules on how to resolve this overlap (for example, by saying that one `impl` is more specific than the other, like in [`min_specialization`](https://github.com/rust-lang/rust/pull/68970)), the type system won't be _coherent_ (i.e. consistent), which is bad in terms of usability and safety.

We already saw that we can define `Scalar<T> * T`, and `BigNum * T`.

Both are useful, and are deemed more useful than `T * Scalar<T>`, which would result in a conflict.

If we limit ourselves to some concrete foreign type (like `u32`), we can still write the following code:

```rust
/// ex1_u32.rs
use std::ops::Mul;

struct Scalar<T>(T);

// For u * s.         (new) ↴
impl<T> Mul<Scalar<T>> for u32
where
    u32: Mul<T, Output = T>, // Don't mind the complicated `where` clause for now.
{
    type Output = T;
    fn mul(self, s: Scalar<T>) -> T { self * s.0 }
}

fn main() {
    let s = Scalar(7u32);
    let u = 6u32;
    println!("{}", u * s);
}
```

```bash
$ rustc ex1_u32.rs
$ ./ex1_u32
42
```

Looking back, this is what the error message was telling us:

> error[E0210]: type parameter `T` must be covered by another type when it appears before the first local type (`Scalar<T>`)

So in our initial compiling example (`impl<T> Mul<T> for Scalar<T>`) the order of types is `[Scalar<T>, T]`, 
so `T` is covered by another type (`Scalar`) before its "uncovered" appearance.

In our failed attempt (`impl<T> Mul<Scalar<T>> for T {}`) the order of types was `[T, Scalar<T>]`, 
so `T` (uncovered) appears before the first local type (`Scalar`).

Now, the order of types is `[u32, Scalar<T>]`, so we don't have an uncovered `T` at all.

Note that the covering type doesn't need to be local, 
so `impl<T> Mul<Scalar<T>> for Vec<T> {}` with the order `[Vec<T>, Scalar<T>]` will compile,
but there needs to be at least one local type.

Let's look at one more complicated example, before [summarizing](#why-cont) everything we learned.

### A more complicated example

We are going to need a trait with two generic type parameters:

```rust
/// ex6_lib.rs

pub trait MulMul<T, U> {}
```

Because we want this to be a foreign trait, we need to build it in a separate crate:

```bash
$ rustc --crate-type=lib ex6_lib.rs
$ ls lib*
libex6_lib.rlib
```

Let's use it:

```rust
/// ex6_order.rs
use ex6_lib::MulMul;

struct Scalar<T>(T);

// Works. Order is [u32, Scalar<T>, T].
impl<T> MulMul<Scalar<T>, T> for u32 {}

// Doesn't Work. Order is [u32, T, Scalar<T>].
impl<T> MulMul<T, Scalar<T>> for u32 {}

fn main() {}
```

Looking up the needed arguments, we arrive at:

```bash
$ rustc ex6_order.rs --edition 2021 --extern ex6_lib=libex6_lib.rlib
...
9 | impl<T> MulMul<T, Scalar<T>> for u32 {}
  |      ^ type parameter `T` must be covered by another type when it appears before the first local type (`Scalar<T>`)
  = note: in this case, 'before' refers to the following order: `impl<..> ForeignTrait<T1, ..., Tn> for T0`, where `T0` is the first and `Tn` is the last
```

Yep, the order matters, and it goes according to the error message.


## Why, cont.

So, Rust want the type system to be coherent, which seems reasonable enough.

But there's one thing that's missing from the definition above:

One option to achieve this would be to let anyone define whatever `impl`s they want,
and if (and only if) there's an actual conflict during compilation, abort with an error.

We saw above that we can `impl<T> Mul<Scalar<T>> for u32` (order of types is `[u32, Scalar<T>]`, so `T` is covered by a local type).
But, this should also allow us to `impl<T> Mul<Scalar<T>> for BigNum`:

```rust
use std::ops::Mul;

use bignum::BigNum;

struct Scalar<T>(T);

// For BigNum * s.
impl<T> Mul<Scalar<T>> for BigNum
where
    u32: Mul<Output = u32> + Mul<T, Output = T>,
{
    type Output = T;
    fn mul(self, s: Scalar<T>) -> T { todo!() }
}
```

```bash
$ rustc --crate-type=lib bignum_lib.rs # has an `impl<T> Mul<T> for BigNum { .. }`.
$ rustc use_bignum.rs --edition 2021 --extern bignum=libbignum_lib.rlib
error[E0119]: conflicting implementations of trait `std::ops::Mul<Scalar<_>>` for type `bignum_lib::BigNum`
 --> use_bignum.rs:8:1
  |
8 | impl<T> Mul<Scalar<T>> for BigNum
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: conflicting implementation in crate `bignum_lib`:
          - impl<T> Mul<T> for BigNum;
```

I guess this is expected since they are, after all, conflicting.

But this error is very different, and it'll go away if remove the `Mul` implementation from `BigNum`:

```diff
-impl<T> Mul<T> for BigNum {
-    type Output = BigNum;
-    fn mul(self, _: T) -> Self::Output { BigNum {} }
-}
```

```bash
$ rustc --crate-type=lib bignum_lib.rs # no Mul impl now.
$ rustc use_bignum.rs --edition 2021 --extern bignum=libbignum_lib.rlib
$ 
```

But why?

The key point is that Rust also wants to **enforce coherence** in a way that:

1. Upstream crates (defining the original traits and types) can add `impl`s without **accidentally** breaking downstream crates.
2. Downstream crates (users of foreign traits and types) can extend them without **accidentally** breaking.
3. Apps can use different libraries which depend on the same upstream crates without fear of "random" breaking because of unexpected interactions between libraries.

These requirements lends us to the actual problem with coherence:

> .. due to coherence, the ability to define impls is a zero-sum game: 
> every impl that is legal to add in a child crate is also an impl that a parent crate cannot add without fear of breaking downstream crates.
>
> ~ [RFC 1023](https://github.com/rust-lang/rfcs/pull/1023)

Rust achieves this by specifying rules around who can define what `impl`s for foreign traits,
and therefore has to balance what `impl`s are more useful for users (or `rebalance` in RFC 1023, or `re-rebalance` in RFC 2451).

In our example, `impl<T> Mul<Scalar<T>> for T { .. }` is fine "in a vacuum", 
but Rust says "you can't implement this because we pinky-promised someone else **they can always implement this other thing** _and_ **it won't break anyone else**".

We'll explore this in a bit by creating a cursed `rustc` that is less strict.

## Cheat sheet

So:

1. If you **define** a type or trait:
    - You can define specific `impl`s for it without fear of breaking users (`impl Mul<u32> for BigNum`, `impl LocalTrait for u32`).
    - Only when you define "very general" `impl`s (sometimes called _blanket impls_) (`impl<T> Mul<T> for BigNum`, `impl<T> LocalTrait for T`) it is a breaking change (and only **direct** users might get the `E0119 conflicting implementations` error, which they can always fix by removing their implementation).
        - The technical definition of "very general" would be the opposite of the order of uncovered type parameters we saw above.
2. If you **use** a foreign trait:
    - You can implement it for local and foreign types, even with generic type parameters, as long as:
        - There's a local type involved
        - If there's an "uncovered" type parameter, it obeys the order we saw above.
3. If you build an app using libraries with common dependencies, **your app never breaks**.

The compiler is going to make sure this holds by enforcing (2), 
limiting the ways crates can **use** foreign traits so 
that the ecosystem can be more usable as a whole.

Let's now look at how `rustc` does this!

_You really should try this at home! Hacking on the compiler is super interesting (and is also a bit of a power trip)._

## How

First things first,

```bash
$ git clone git@github.com:rust-lang/rust.git
```

The `rust-lang/rust` repo is a bit more than 2.5 million lines of Rust.

```bash
cd rust
$ cargo install tokei
$ tokei
===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
...
-------------------------------------------------------------------------------
 Rust                26673      2416621      1949322       189567       277732
 |- Markdown          4331       237577         9147       180760        47670
 (Total)                        2654198      1958469       370327       325402
```

We can use the guide to navigate around, but it's going to be easier to ripgrep though.

Let's look for the code checking and generating this error.

In this case, a good unique string is probably `"for T0"` in `*.rs` files.

Indeed, we find only two results, one of which is the correct `compiler/rustc_hir_analysis/src/coherence/orphan.rs` file.

It's in a function called `emit_orphan_check_error`, called from `do_orphan_check_impl` 
(the "orphan" in this case is "_an `impl` that is implementing a trait you don't own for a type you don't own_", according to the RFCs).

This sounds promising.

Now, I'm not going to pretend to be some compiler expert, I just hacked on the compiler a few times a while ago.
We are just exploring the sources together, pointing at things and making observations. This is more a garden stroll than a botanical lecture.

The relevant bit of `emit_orphan_check_error` looks like this:

(code is slightly simplified and I added the `//!` comments, the full code is [here](https://github.com/rust-lang/rust/blob/master/compiler/rustc_hir_analysis/src/coherence/orphan.rs)):

```rust
match err {
    //! `param_ty` the the type parameter (`T`).
    traits::OrphanCheckErr::UncoveredTy(param_ty, Some(local_type)) => {
        //! Find the span (location in the sources) of the generic param (the `T` in `impl<T, U, ..> ..`).
        let mut sp = sp;
        for param in generics.params {
            if param.name.ident().to_string() == param_ty.to_string() {
                sp = param.span;
            }
        }
        

        //! Emit an error that points to the span (`sp`) of the `T` parameter.
        struct_span_err!(
            tcx.sess,
            sp,
            E0210,
            //! Format the actual error message to include the names.
            "type parameter `{}` must be covered by another type \
            when it appears before the first local type (`{}`)",
            param_ty,
            local_type
        )
        //! After generating the basic error, we can add labels to other spans.
        //! (in this case, just repeat what we said before, but we could point to other spans if we choose)
        .span_label(
            sp,
            format!(
                "type parameter `{}` must be covered by another type \
            when it appears before the first local type (`{}`)",
                param_ty, local_type
            ),
        )
        //! We can add a note or even a `multipart_suggestion` for actual suggestions.
        .note(
            "in this case, 'before' refers to the following order ...",
        )
        .emit();
    }
}
```

Pretty cool!

Now for the actual check. `do_orphan_check_impl` calls `traits::orphan_check`, where we find our first specimen: `orphan_check_trait_ref`.

The name (and signature, and docs) look quite promising: it takes a reference to a trait and a crate (well, not a crate but close enough) and returns a result.

```rust
/// Checks whether a trait-ref is potentially implementable by a crate.
///
/// The current rule is that a trait-ref orphan checks in a crate C:
/// ... (many many lines of documentation follows)
#[instrument(level = "trace", ret)]
fn orphan_check_trait_ref<'tcx>(
    trait_ref: ty::TraitRef<'tcx>,
    in_crate: InCrate,
) -> Result<(), OrphanCheckErr<'tcx>> {
    let mut checker = OrphanChecker::new(in_crate);
    match trait_ref.visit_with(&mut checker) {       //! This is the main part. 
        ControlFlow::Continue(()) => Err(OrphanCheckErr::NonLocalInputType(checker.non_local_tys)),
        ControlFlow::Break(OrphanCheckEarlyExit::ParamTy(ty)) => {
            // Does there exist some local type after the `ParamTy`.
            checker.search_first_local_ty = true;
            if let Some(OrphanCheckEarlyExit::LocalTy(local_ty)) =
                trait_ref.visit_with(&mut checker).break_value()
            {
                Err(OrphanCheckErr::UncoveredTy(ty, Some(local_ty)))
            } else {
                Err(OrphanCheckErr::UncoveredTy(ty, None))
            }
        }
        ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(_)) => Ok(()),
    }
}
```

But where's the actual code? We need to go to the `OrphanChecker` struct to actually find (and modify!) the rules.

We can already see there's quite a bit of complexity associated with generating a better error message:
The first call to `visit_with` returns with a value that already indicates an error, but the function reruns `visit_with` to try and find the first local type, which is only used when generating the error.

What we expect to find when we look at `OrphanChecker` is some sort of [Visitor](https://en.wikipedia.org/wiki/Visitor_pattern)-style pattern, where we can recursively travel the generic parameters of the trait ref in-order.

It should skip all non-local types, and either:

1. Find a local type and bail (triggering the `ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(_)) => Ok(())` "good" branch) or
2. Find an uncovered generic type parameter (triggering the `ControlFlow::Break(OrphanCheckEarlyExit::ParamTy(ty) => { ... }` error branch) or
3. Find nothing, meaning there are only non-local types (triggering the `ControlFlow::Continue(()) => Err(OrphanCheckErr::NonLocalInputType(...)` error branch)

The fact that the usage is `trait_ref.visit_with(&mut checker)` indicates that this will probably be done using a common trait (which make sense: the compiler probably does this type of traversal a lot).

Now that we are mentally ready, we can dive into the `OrphanChecker` code, skipping right to the visitor implementation.

The complete code also deals with a bunch of unstable features which I'm going to elide here, and as before `//!` are added for clarity.

```rust
//! 'tcx is the common lifetime for all data associated with type-checking.
enum OrphanCheckEarlyExit<'tcx> {
    //! An uncovered generic type parameter `T`.
    ParamTy(Ty<'tcx>),
    LocalTy(Ty<'tcx>),
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for OrphanChecker<'tcx> {
    type BreakTy = OrphanCheckEarlyExit<'tcx>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        let result = match *ty.kind() {
            //! If we found an (uncovered) generic type parameter first, break and return it.
            //! (Unless we already failed and are just looking for the first local type for the error)
            ty::Param(..) => if self.search_first_local_ty {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(OrphanCheckEarlyExit::ParamTy(ty))
            },
            
            //! Handles reference types like `&SomeType` and `&mut SomeType`.
            //! Here, `ty` is the `SomeType` in both examples.
            //! References are not covering the type inside them, 
            //! so we ignore them and traverse the inner type.
            ty::Ref(_, ty, _) => ty.visit_with(self),

            //! `Adt`s (Algebraic data types) are structs, enums and unions,
            //! so this handles `impl ForeignTrait for SomeStruct<T>`, `impl ForeignTrait<OtherStruct<T>> for SomeStruct<T>` and so on.
            //! The `def` is the ADT definition.
            //! The `substs` (short for substitutions) are the generic parameters of the ADT.
            //! For `SomeStruct<T>` the substs will be `T` (so a generic), but for `SomeStruct<u32>` they will be the `u32`.
            ty::Adt(def, substs) => {
                //! The actual ADT is is local, we don't care about the substitutions.
                if self.def_id_is_local(def.did()) {
                    ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty))
                //! Fundamentals aren't covering their substitutions.
                //! For example, `Box` is fundamental, `Pin` is also fundamental.
                //! So `Box<T>` should be treated as `T`.
                //! Like in `ty::Ref`, ignore the outer (fundamental) type and traverse the substitutions.
                } else if def.is_fundamental() {
                    substs.visit_with(self)
                //! Continue the search for a local type.
                //! Note that even if the type is foreign (like `Vec` in `Vec<T>`), we don't traverse the substitutions.
                //! Because the substitutions are ignored, 
                //! the `ty::Param(..)` branch won't be called, meaning that `T` is considered "covered".
                } else {
                    ControlFlow::Continue(())
                }
            }

            //! All the builtins are non local, but they also cover `T`.
            //! (So [T] is considered covered)
            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Str
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::Array(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::Never
            | ty::Tuple(..)
            | ty::Alias(ty::Projection, ..) => ControlFlow::Continue(()),

            //! ..., Handling other stuff like ty::Foreign(def_id), ty::Dynamic(tt, ..), ty::Alias(ty::Opaque, ..) and ty::Closure(did, ..).
        };

        result
    }

    fn visit_region(&mut self, _r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> { ... }

    fn visit_const(&mut self, _c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> { ... }
}
```

Ok. Let's take this for a spin.


### Setting up, building & playing around with rustc

```
$ ./x.py setup
Building bootstrap
...
Welcome to the Rust project! What do you want to do with x.py?
a) library: Contribute to the standard library
b) compiler: Contribute to the compiler itself
c) codegen: Contribute to the compiler, and also modify LLVM or codegen
...
```

We are going to build the compiler itself, so let's go with option `b`.

Now we can build the compiler, using

```
$ ./x.py build library
... lots and lots of stuff ...
Build completed successfully in 0:02:54
```

Wait, what? Weren't we building the compiler? Why are we building `library`?

The [guide](https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run.html) explains this (stage0 is the latest prebuilt beta compiler):

> This may _look_ like it only builds the standard library, but that is not the case. What this command does is the following:
> - Build std using the stage0 compiler
> - Build rustc using the stage0 compiler
>   - This produces the stage1 compiler
> - Build std using the stage1 compiler
> - This final product (stage1 compiler + libs built using that compiler) is what you need to build other Rust programs.

Cool! Let's make sure we actually have working compiler.

First, we'll link this compiler to our `rustup`:

```
$ rustup toolchain link stage1 build/host/stage1
```

Now, we can use it to build our little example code:

```
$ rustc +stage1 ex1.rs
error[E0210]: type parameter `T` must be covered by another type when it appears before the first local type (`Scalar<T>`)
...
```

Cool! But not terribly exiting (`cargo +stage1 ..` can be used as well).

Can we create a cursed version of the compiler which ignores the orphan rules?

```diff
@@ -582,6 +582,10 @@ fn orphan_check_trait_ref<'tcx>(
     trait_ref: ty::TraitRef<'tcx>,
     in_crate: InCrate,
 ) -> Result<(), OrphanCheckErr<'tcx>> {
+    if true {
+        return Ok(());
+    }
+
```

```bash
$ ./x.py build library
...
   Compiling rustc-main v0.0.0 (/Users/oravid/writing_workspace/rust/compiler/rustc)
    Finished release [optimized] target(s) in 9.34s
Assembling stage1 compiler
Building stage1 library artifacts (aarch64-apple-darwin)
...
   Compiling core v0.0.0 (/Users/oravid/writing_workspace/rust/library/core)
...
error[E0119]: conflicting implementations of trait `IntoIterator` for type `[_; _]`
   --> library/core/src/iter/traits/collect.rs:266:1
    |
266 | impl<I: Iterator> IntoIterator for I {
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ conflicting implementation for `[_; _]`
    |
   ::: library/core/src/array/iter.rs:44:1
    |
44  | impl<T, const N: usize> IntoIterator for [T; N] {
    | ----------------------------------------------- first implementation here
    |
    = note: downstream crates may implement trait `iter::traits::iterator::Iterator` for type `[_; _]`
...
```

Oh no. Searching for `orphan_check_trait_ref`, we can see it is used in 

```rust
pub fn trait_ref_is_knowable<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), Conflict> {
    //! ...
    if orphan_check_trait_ref(trait_ref, InCrate::Remote).is_ok() {
        // A downstream or cousin crate is allowed to implement some
        // substitution of this trait-ref.
        return Err(Conflict::Downstream);
    }
    //! ...
    if orphan_check_trait_ref(trait_ref, InCrate::Local).is_ok() {
        Ok(())
    } else {
        Err(Conflict::Upstream)
    }
}
```

Which also explains the `in_crate` parameter from before.

Let's do something even more cursed. What if we make `L` a special type parameter name that isn't considered uncovered by the orphan checker?

```diff
-  ty::Param(..) => if self.search_first_local_ty {
+  ty::Param(p) => if self.search_first_local_ty || p.name.as_str() == "L" {
       ControlFlow::Continue(())
   } else {
       ControlFlow::Break(OrphanCheckEarlyExit::ParamTy(ty))
   },
```

(The full diff is [here](https://github.com/rust-lang/rust/compare/master...ohadravid:rust:cursed-coherence))


Let's rebuild the compiler (using `./x.py build library` again), and check if it works as expected:

```bash
$ ./x.py build library
...
Build completed successfully in 0:02:54
```

```bash
$ rustc +stage1 ex1.rs
error[E0210]: type parameter `T` must be covered by another type when it appears before the first local type (`Scalar<T>`)
```

Good. Let's replace `T` with `L` for the second `impl`:

```rust
/// ex1_cursed.rs
use std::ops::Mul;

#[derive(Clone, Copy)]
struct Scalar<T>(T);

// For s * n.
impl<T: Mul<Output = T>> Mul<T> for Scalar<T> {
    type Output = T;
    fn mul(self, s: T) -> T { self.0 * s }
}

// For n * s. L for local.
impl<L: Mul<Output = L>> Mul<Scalar<L>> for L {
    type Output = L;
    fn mul(self, s: Scalar<L>) -> L { self * s.0 }
}

fn main() {
    let s = Scalar(7u32);
    let n = 6u32;
    println!("{}", s * n);
    println!("{}", n * s);
}
```

```bash
$ rustc +stage1 ex1_cursed.rs
$ ./ex1_cursed
42
42
```

Huzzah!

But... if it works, why is this bad?

Let's try to use both `BigNum` and `Scalar` as libraries in a new binary:

```rust
/// usage.rs
use bignum;
use cursed;

fn main() {
    let b = bignum::BigNum {};
    let s = cursed::Scalar(bignum::BigNum {});
    let r: bignum::BigNum = b * s;
}
```

```bash
$ # note: both libraries are completely independent. 
$ rustc +stage1 --crate-type=lib --edition 2021 ./ex1_cursed.rs
$ rustc +stage1 --crate-type=lib --edition 2021 ./bignum_lib.rs
$
$ rustc +stage1 usage.rs --edition 2021 --extern bignum=libbignum_lib.rlib --extern cursed=libex1_cursed.rlib
error[E0284]: type annotations needed: 
...
7 |     let r: bignum::BigNum = b * s;
  |                               ^ cannot satisfy `<BigNum as Mul<Scalar<BigNum>>>::Output == BigNum`
```

The compiler can't choose between the two overlapping implementations,
and the result is that using both libraries together will cause a compilation error, 
but only for the unlucky application that happened to use both crates.

This is exactly what the RFCs wanted to avoid: we want a safe, extendable and composable crate ecosystem,
and the price of forbidding a few `impl`s is well worth it.

## Summary

We stared with a compilation error, and wanted to dig deeper.

```rust
error[E0210]: type parameter `T` must be covered by another type when it appears before the first local type (`Scalar<T>`)
 --> ex1.rs:9:6
  |
9 | impl<T: Mul<Output = T>> Mul<Scalar<T>> for T {}
  |      ^ type parameter `T` must be covered by another type when it appears before the first local type (`Scalar<T>`)
  |
  = note: implementing a foreign trait is only possible if at least one of the types for which it is implemented is local, and no uncovered type parameters appear before that first local type
  = note: in this case, 'before' refers to the following order: `impl<..> ForeignTrait<T1, ..., Tn> for T0`, where `T0` is the first and `Tn` is the last
```

After trying out a few non-compiling examples and reviewing the RFCs, we figured out *why* this example wasn't compiling:
If it were, we could be breaking apps that use our code because of potential conflicts with other libraries, 
and it's important to Rust that it's easy (relatively speaking) to reason about what changes to a library are considered breaking changes.

We went to the source and found the actual implementation of this error and the checks that generate it.

```rust
struct_span_err!(
    tcx.sess,
    sp,
    E0210,
    "type parameter `{}` must be covered by another type \
    when it appears before the first local type (`{}`)",
    param_ty,
    local_type
)
```

```rust
enum OrphanCheckEarlyExit<'tcx> {
    ParamTy(Ty<'tcx>),
    LocalTy(Ty<'tcx>),
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for OrphanChecker<'tcx> {
    type BreakTy = OrphanCheckEarlyExit<'tcx>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        let result = match *ty.kind() {
            ty::Param(..) => if self.search_first_local_ty {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(OrphanCheckEarlyExit::ParamTy(ty))
            }

            ...
        }
    }
}
```

We configured and built of our own `rustc` and were even able to build a cursed compiler that accepts our code, showing how unsuspecting users's code will break if this was allowed. 

Not too shabby.

## Outro

The compiler is a big project, but it is much much more approachable than you think.
The whole point of the guide it to help you along! There's even a section on using Git.

Here's the link again:

[Getting Started - Rust Compiler Development Guide](https://rustc-dev-guide.rust-lang.org/getting-started.html#getting-started)

Give it a shot! You'll meet friendly people and friendly rust code 🦀


_Edit, 20230512 - Reworded the opening a bit and changed the title_