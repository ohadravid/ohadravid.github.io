---
title: "Put a Pin on That"
date: 2023-07-11T17:00:00+00:00
tags: ["rust", "async"]
type: post
showTableOfContents: true
image: "/2023-07-put-a-pin-on-that/dalle_a_pushpin_futuristic_drawing.jpg"
weight: 2
---


There are some things you keep learning and forgetting (or perhaps you've never learned them in the first place?).

For me, one of those things is `Pin/Unpin` in Rust.

Every time I [read](https://fasterthanli.me/articles/pin-and-suffering) an [explanation](https://blog.cloudflare.com/pin-and-unpin-in-rust/) about [pinning](https://doc.rust-lang.org/std/pin/index.html), my brain is like 👍, and a few weeks later is like 🤔🤨.

So, I'm writing this as a way to force my brain to retain (pin?) this knowledge. We'll see how it goes!

# Pin

`Pin` is _a type of pointer_, which can be thought of as a middle ground between `&mut T` and `&T`.

The point of `Pin<&mut T>` is to say:

1. ✅ This value can be modified (like `&mut T`) but
2. 🙅 This value cannot be moved (unlike `&mut T`)

Why? Because some values must never be moved, or special care is needed to do so. 

A prime example of this are self-referential data structures. They occur naturally when using `async`, because futures tend to reference their own locals **across await points**.

This seemingly benign future:

```rust
async fn self_ref() {
    let mut v = [1, 2, 3];

    let x = &mut v[0];

    tokio::time::sleep(Duration::from_secs(1)).await;

    *x = 42;
}
```

requires a self referential structure, because under the hood futures are _state machines_ (unlike [closures](https://doc.rust-lang.org/reference/types/closure.html#closure-types)).

Note that `self_ref` passes control back to the caller on the first `await`. This means that even though `v` and `x` _look_ like regular stack variables, something more complex must be going on here. 

The compiler wants to generate something like this:

```rust
enum SelfRefFutureState {
    Unresumed,        // Created and wasn't polled yet.
    Returned,
    Poisoned,         // `panic!`ed.
    SuspensionPoint1, // First `await` point.
}

struct SelfRefFuture {
    state: SelfRefFutureState,
    v: [i32; 3],
    x: &'problem mut i32, // a "reference" to an element of `self.v`, 
                          // which is a big problem if we want to move `self`.
                          // (and we didn't even consider borrowchecking!)
}
```

With `await`ing being an update to the `state` field and running the associated code ([see a full example in the end](#appendix-a---a-hand-rolled-self-referential-future)).

But! You can totally move this future if you tried:

```rust
let f = self_ref();
let boxed_f = Box::new(f); // Evil?

let mut f1 = self_ref();
let mut f2 = self_ref();

std::mem::swap(&mut f1, &mut f2); // Blasphemy?
```

What gives? As a wise compiler once said:

> futures do nothing unless you `.await` or poll them<br>
> `#[warn(unused_must_use)]` on by default
>
> -- <cite>rustc</cite>

This is because calling `self_ref` really does nothing: we actually get back something more like this[^0]: 

```rust
struct SelfRefFuture {
    state: SelfRefFutureState,
    v: MaybeUninit<[i32; 3]>,
    x: *mut i32, // a pointer into `self.v`, 
                 // still a problem if we want to move `self`, but only after it is set.
    //
    // .. other locals, like the future returned from `tokio::time::sleep`.
}
```

which can be moved **safely**[^1] in it's initial (`Unresumed`) state.

```rust
impl SelfRefFuture {
    fn new() -> Self {
        Self {
            state: SelfRefFutureState::Unresumed,
            v: MaybeUninit::uninit(),
            x: std::ptr::null_mut(),
            // ..
        }
    }
}
```

Only when we start polling on `f` we get into the self-ref problem (once the `x` pointer is set), 
and if `f` is wrapped in a `Pin` all those moves become `unsafe`, which is exactly what we want.

Because a lot of futures shouldn't be moved around in memory once they "start", they can only be worked with safely if they are wrapped in a `Pin`, and so async-related functions tend to accept a `Pin<&mut T>` (Assuming they don't need to move the value).

[^0]: The actual layout of generators is [more complex in practice](https://github.com/rust-lang/rust/blob/master/compiler/rustc_mir_transform/src/generator.rs).

[^1]: In fact, all **types** can be moved ([`mem::swap`](https://doc.rust-lang.org/std/mem/fn.swap.html) accepts any type `T`). **Values**, however, can be wrapped in a `Pin`, which enforces the rules above (getting a `&mut T` from `Pin<&mut T>` is `unsafe` in the general case).

## A tiny example

Here, no pinning is required:

```rust
use tokio::time::timeout;

async fn with_timeout_once() {
    let f = async { 1u32 };

    let _ = timeout(Duration::from_secs(1), f).await;
}
```

But if we want to call `timeout` multiple times (for example, because we want to retry) we'll have to use `&mut f` (or we'll get `use of moved value`), which is going to cause the compiler to complain about pinning:

```rust
use tokio::time::timeout;

async fn with_timeout_twice() {
    let f = async { 1u32 };

    // error[E0277]: .. cannot be unpinned, consider using `Box::pin`.
    //               required for `&mut impl Future<Output = u32>` to implement `Future`
    let _ = timeout(Duration::from_secs(1), &mut f).await;
    
    // An additional retry.
    let _ = timeout(Duration::from_secs(1), &mut f).await;
}
```

Why?

Because a few levels down, `timeout` is calling `Future::poll` which is [defined](https://doc.rust-lang.org/std/future/trait.Future.html#tymethod.poll) as 

```rust
fn poll(self: Pin<&mut Self>, ...) -> ... { ... }
```

When we `await`ed on `f` itself, we gave up ownership on it.

This allowed the compiler to handle the pinning for us, but it can't do that if we only provide a `&mut f`, since we could easily break `Pin`'s invariants:

```rust
use tokio::time::timeout;

async fn with_timeout_twice_with_move() {
    let f = async { 1u32 };

    // error[E0277]: .. cannot be unpinned, consider using `Box::pin`.
    let _ = timeout(Duration::from_secs(1), &mut f).await;

    // .. because otherwise, we could move `f` to a new memory location, after it was polled!
    let f = *Box::new(f);

    let _ = timeout(Duration::from_secs(1), &mut f).await;
}
```

So **we** don't care about pinning, and **our** future is not really special in any way (or is it? more on that later!), and we don't move our future anywhere, but we are using an API which also allows for futures that **are** special, and so we need to play along by `pin!`ing our future:

```rust
use tokio::pin;
use tokio::time::timeout;

async fn with_timeout_twice() {
    let f = async { 1u32 };

    pin!(f);  // f is now a `Pin<&mut impl Future<Output = u32>>`.
    
    let _ = timeout(Duration::from_secs(1), &mut f).await;
    let _ = timeout(Duration::from_secs(1), &mut f).await;
}
```

This is sort of like the beloved

```
expected `&u32`, found `u32`
help: consider borrowing here: `&1u32`
```

with a bit more steps, traits, opaque types and macros.

We do need those extra steps: creating a `Pin<&mut T>` requires a little more effort because we also need to make sure that no `&mut T` is left around or can be obtained later (like we saw above) which would defeat the purpose of the `Pin`.

This leads us to a more accurate phrasing of the no-move rule: the pointed-to value must not move until *the value* is dropped (regardless of when the `Pin` is dropped!).

That's the job of the `pin!` macro: it makes sure that the original `f` is no longer visible to our code, thus enforcing `Pin`'s invariants (we can't move it if we can't see it).

Tokio's `pin!` [implementation](https://docs.rs/tokio/latest/src/tokio/macros/pin.rs.html#125-144) expands `pin!(f)` to this:

```rust
// Move the value to ensure that it is owned
let mut f = f;
// Shadow the original binding so that it can't be directly accessed
// ever again.
#[allow(unused_mut)]
let mut f = unsafe {
    Pin::new_unchecked(&mut f)
};
```

The standard library's version of `pin!` is a bit [cooler](https://doc.rust-lang.org/stable/src/core/pin.rs.html#1244), but the same reasoning is used: shadow the original value with a newly created `Pin` so it can no longer be accessed and moved.

## A 📦

So `Pin` is a (zero-sized wrapper around another) pointer, and it's a bit like `&mut T` with more rules.

The next problem is going the be "returning borrowed data".

We can't return the pinned future from before:

```rust
use std::future::Future;

async fn with_timeout_and_return() -> impl Future<Output = ()> {
    let f = async { 1u32 };

    pin!(f);  // f is now a `Pin<&mut impl Future<Output = u32>>`.

    let s = async move {
        let _ = timeout(Duration::from_secs(1), &mut f).await;
    };

    // error[E0515]: cannot return value referencing local variable `f`
    s
}

```

It should be more clear why now: the pinned `f` is now a pointer,
and it points to data (the async closure) that won't be there once we return from the function.

We therefore can use `Box::pin`[^2]:

```diff
-pin!(f);
+let mut f = Box::pin(f);
```

Making `f` a `Pin<Box<impl Future<Output = u32>>`.

But didn't we just say that `Pin<&mut T>` is a (wrapper around a) pointer "in between" `&mut T` and `&T`?

Well, a `mut Box<T>` is also like a `&mut T`, but with ownership.

So a `Pin<Box<T>>` is a pointer "in between" a mutable `Box<T>` and an immutable `Box<T>`, with the same exceptions (the value can be modified but cannot be moved).

[^2]: In this example, we can also delay the pinning, moving the original future into the `async` block and `pin` it there.

# Unpin

`Unpin` is a trait. It's not "the opposite" of `Pin`, because `Pin` is a type of pointer and traits (however good their marketing) cannot be the opposite of pointers.

`Unpin` is also an auto-trait (the compiler implements it for you automatically when possible), marking a type whose values **can** be moved after being pinned (so for example, it will not be self-referential).

The main point is that if `T: Unpin`, we can always [`Pin::new`](https://doc.rust-lang.org/std/pin/struct.Pin.html#method.new) and `Pin::{into_inner,get_mut}` values of `T`, meaning we can easily go from and to a "regular" mutable value and ignore the associated complexities with working directly with pinned values.

`Unpin` is also why `Box::pin` is so useful: It (or rather, [`Box::into_pin`](https://doc.rust-lang.org/std/boxed/struct.Box.html#method.into_pin)) can safely call the unsafe `Pin::new_unchecked` because "it is not possible to move or replace the insides of a `Pin<Box<T>>` when `T: !Unpin`", and the resulting `Box` [is always `Unpin`](https://doc.rust-lang.org/std/boxed/struct.Box.html#impl-Unpin-for-Box%3CT,+A%3E) because moving it doesn't move the actual value.

## Another tiny example

We can create an `Unpin` future by hand:

```rust
fn not_self_ref() -> impl Future<Output = u32> + Unpin {
    struct Trivial {}

    impl Future for Trivial {
        type Output = u32;

        fn poll(self: Pin<&mut Self>, _cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
            std::task::Poll::Ready(1)
        }
    }

    Trivial {}
}
```

Now, we can call `timeout` on it multiple times without pinning:

```rust
async fn not_self_ref_with_timeout() {
    let mut f = not_self_ref();

    let _ = timeout(Duration::from_secs(1), &mut f).await;
    let _ = timeout(Duration::from_secs(1), &mut f).await;
}
```

Any future which is created using the `async fn` or `async {}` syntax is considered `!Unpin` - meaning that once we put it inside a `Pin`, we won't be able to take it out again.

# Summary

- `Pin` is a wrapper around another pointer which is a bit like `&mut T`, with the additional rule that it is unsafe to move the value it points to until the *value* is dropped.
- To safely work with self-ref structures, we must prevent them from moving once we set a self-referential field.
- Placing a value inside a `Pin` does exactly that.
- Constructing a `Pin` is harder because `Pin` promise that moving is impossible for the lifetime of the value, so we can't create it without giving up the ability to create a `&mut T` later on and breaking the `Pin`'s invariants.
- When `await`ing on an owned `Future`, the compiler can handle the pinning because it can know that the `Future` won't move once ownership is transferred.
- Otherwise, **we** need to handle the pinning (for example with `pin!` or `Box::pin`) which is a bit tricky because of all this.
- `Unpin` is a marker trait that says that a type **can** be safely moved even after it was wrapped in a `Pin`, making everything simpler.
- Most structures are `Unpin`, but `async fn` and `async {}` always generate `!Unpin` structures.

## Appendix A - A hand rolled self-referential `Future`

_Note: I'm not an `unsafe` lawyer.  This code is indented for educational purposes about `async`, so there might be some glaring UB issues I missed._

We will not use `MaybeUninit`s here so we can focus only on the `unsafe` operations w.r.t `Pin`, 
but IRL the compiler won't use `Option` (or initialize `v` twice, etc) as it is unneeded & slower.

```rust
enum SelfRefFutureState {
    Unresumed,        // Created and wasn't polled yet.
    SuspensionPoint1, // First `await` point.
    Returned,
    Poisoned,         // `panic!`ed.
}

struct SelfRefFuture {
    state: SelfRefFutureState,
    v: [i32; 3],
    x: *mut i32, // a pointer into `self.v`,
                 // a problem if `self` moves, but only after it is set.
    sleep: Option<tokio::time::Sleep>,

    // Ensure the we are !Unpin.
    _m: std::marker::PhantomPinned,
}

impl SelfRefFuture {
    fn new() -> Self {
        Self {
            state: SelfRefFutureState::Unresumed,
            v: [0; 3],
            x: std::ptr::null_mut(),
            sleep: None,
            _m: std::marker::PhantomPinned,
        }
    }
}

impl Future for SelfRefFuture {
    type Output = ();

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // Safety: We aren't going to move `self`, promise.
        let this = unsafe { self.get_unchecked_mut() };

        match this.state {
            SelfRefFutureState::Unresumed => {
                this.v = [1, 2, 3];
                this.x = this.v.as_mut_ptr().wrapping_add(1);
                this.sleep = Some(tokio::time::sleep(Duration::from_secs(1)));
                this.state = SelfRefFutureState::SuspensionPoint1;

                // Safety: We are the owners of `sleep`, and we aren't moving it.
                let pinned_sleep = unsafe { Pin::new_unchecked(this.sleep.as_mut().unwrap()) };
                Future::poll(pinned_sleep, cx)
            }
            SelfRefFutureState::SuspensionPoint1 => {
                // Safety: Same as above.
                let pinned_sleep = unsafe { Pin::new_unchecked(this.sleep.as_mut().unwrap()) };

                if let std::task::Poll::Pending = Future::poll(pinned_sleep, cx) {
                    return std::task::Poll::Pending;
                };

                // Safety: We initialized `v` and `x` before moving to this state,
                // No one else can move us because `Self` is wrapped in a `Pin`,
                // so `x` is still valid.
                unsafe { this.x.write(42) };
                this.state = SelfRefFutureState::Returned;

                std::task::Poll::Ready(())
            }
            SelfRefFutureState::Returned => std::task::Poll::Ready(()),
            SelfRefFutureState::Poisoned => {
                panic!()
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let f = SelfRefFuture::new();
    f.await;
}
```
