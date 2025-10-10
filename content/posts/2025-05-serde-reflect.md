---
title: "A Rust API Inspired by Python, Powered by Serde"
summary: "Years ago, I worked on reimplementing some Python code in Rust and needed to adapt Python’s dynamic reflection capabilities (aka `__getattr__`) to the strict and compiled world of Rust..."
date: 2025-05-07T12:00:00+00:00
tags: ["rust", "python", "api", "design"]
type: post
showTableOfContents: true
image: "/2025-05-serde-reflect/rick_and_motry_20min_adventure_with_types.webp"
weight: 2
---

Years ago, I worked on reimplementing[^1] some Python code in Rust ([yet again]({{< ref "/posts/2023-03-rusty-python.md" >}})),
and this time I needed to adapt Python’s dynamic capabilities (aka `__getattr__`) to the strict and compiled world of Rust.

After some deliberation, and armed with the `serde` (de)serialization crate, I set out to do just that.

So if you ever wondered what tricks Rust can learn from Python, and if `serde` can be used for reflection (please go along with me here), then the answer is very much yes! \
Let me show you how, and also why.

_Note: I'll try to keep this somewhat approachable to Python programmers with only passing experience in Rust, it'll be fun._ 🦀🐍

_I also gave a talk on this topic at EuroRust 2025, you can find the slides [here](/2025-05-serde-reflect/eurorust_2025_serde_driven_reflection.pdf)._

<img
  src="/2025-05-serde-reflect/rick_and_motry_20min_adventure_with_types.webp"
  alt="Rick and Morty - Let's go. In and out. 20 minutes adventure, with complex Rust code with many types in the wormhole"
  title="Let's go. Python in Rust out. 20 minutes adventure."
  width="1280"
  height="720"
/>

[^1]: The result of this work is the [`wmi-rs` crate], which I maintain to this day.

[`wmi-rs` crate]: https://github.com/ohadravid/wmi-rs

## A Python Inspiration

Using Python (on Windows), there’s a [magical package](https://timgolden.me.uk/python/wmi/tutorial.html) that can be used to get all sorts of information about the system.

Without going into too much detail, it boils down to this: Say you want to list all the installed [physical fans](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-fan). You can do that using the following code:

```python
import wmi
c = wmi.WMI()
for fan in c.Win32_Fan():
    if fan.ActiveCooling:
        print(f"Fan `{fan.Name}` is running at {fan.DesiredSpeed} RPM")
```

This uses a couple of clever `__getattr__` implementations, and ends up running this equivalent code behind the scenes:

```python
for fan in c.query("SELECT * FROM Win32_Fan"):
    if fan.wmi_property("ActiveCooling").value is True:
        print(f"Fan `{fan.wmi_property('Name').value}` is running at {fan.wmi_property('DesiredSpeed').value} RPM")
``` 

(The `wmi_property` method returns another object which holds the final value as well as its type.)

Putting aside [what `WMI` even is](https://learn.microsoft.com/en-us/windows/win32/wmisdk/example--getting-wmi-data-from-the-local-computer), this is the kind of thing Python is extraordinarily good at:
A clear, concise, and intuitive interface abstracting over a complex and gnarly implementation.

In Python, you need but override a few magic methods to implement this, but Rust is a completely different beast.

What can we do if we want to create a Rust crate that provides a nice API to users, similar to the one above?

## What We Are Abstracting Over

It'll help to start by expressing the "raw" API in pure Rust. It's raw in the sense that we can use it to get the _raw data_ from the system, but nothing more.

We have two types: the `Value` type that can hold different types of values, 
and the `Object` type that provides access to named attributes (which are themselves `Value` instances).

We end up with something like this (the full code is available [on GitHub](https://github.com/ohadravid/serde-reflect)):

```rust
mod raw_api {
    pub struct Object { .. }

    pub enum Value {
        Bool(bool),
        I1(i8),
        // ..
        UI8(u64),
        String(String),
        Object(Object),
    }

    impl Object {
        pub fn get_attr(&self, name: &str) -> Value { .. }
    }

    pub fn query(query: &str) -> Vec<Object> { .. }
}
```

Simple, but rather painful for the user to work with:

```rust
let res = raw_api::query("SELECT * FROM Win32_Fan");

for obj in res {
    if obj.get_attr("ActiveCooling") == Value::Bool(true) {
        if let Value::String(name) = obj.get_attr("Name") {
            if let Value::UI8(speed) = obj.get_attr("DesiredSpeed") {
                println!("Fan `{name}` is running at {speed} RPM");
            }
        }
    }
}
```

Since any field can be of any type, the user must manually check (either with `match` or, as we did, with `if let`) 
what variant of the `Value` enum they got every time they interact with it. \
This is especially cumbersome when one wants to query many different types of objects (`Win32_Battery` 🔋, `Win32_UserAccount` 💁, `Win32_Printer` 🖨️, ...).

<img src="/2025-05-serde-reflect/its_raw.webp" alt="Gordon Ramsay yelling IT'S RAW" width="60%" loading="lazy" />

_Note: this is a simplification of the [underlying API](https://learn.microsoft.com/en-us/windows/win32/api/wbemcli/nf-wbemcli-iwbemclassobject-get) we have to use,
but it's close enough that we can design our higher level API based on it. You can check the [`wmi-rs` crate] source code for the full details._

## A Possible Design

Inspired by the Pythonic API, what if instead we could do something like this:

```rust
// 1. The user defines a custom struct for the type of objects to query.
struct Fan {
    name: String,
    active_cooling: bool,
    desired_speed: u64,
}

// 2. Specify that `query` should to return instances of `Fan`.
let res: Vec<Fan> = api::query();

// 3. Profit.
for fan in res {
    if fan.active_cooling {
        println!("Fan `{}` is running at {} RPM", fan.name, fan.desired_speed);
    }
}
```

Much nicer!

One of the distinct features of Rust is that [generic return types](https://blog.jcoglan.com/2019/04/22/generic-returns-in-rust/) can change the behavior of a function 
(with [`.collect()`](https://doc.rust-lang.org/nightly/std/iter/trait.Iterator.html#method.collect) being the most famous example of this), so how can we implement something like this?

Imagine we went ahead and updated `query` to accept some generic type `T`. 
What trait can we use to constrain `T` so we can implement the function?

```rust
fn query<T>() -> Vec<T> where T: ??? { ??? }
```

The standard library offers [`any::type_name`](https://doc.rust-lang.org/beta/std/any/fn.type_name.html) which could (not really) help us to build the `SELECT` query,
but without resorting to [violence](https://jack.wrenn.fyi/blog/deflect/), we seem to be on our own.

## Crawl Before You Can Walk

Let's start with a relatively simple solution: Let's _define a new trait_ that:

1. Provides the name of the object to query.
2. Handles the construction of e.g. `Fan`s from `Object`s.

This will give us a firm ground to stand on before we [dive deeper](#implementing-a-deserializer).

```rust
trait Queryable {
    fn object_name() -> &'static str;
    fn from(obj: Object) -> Self;
}

fn query<T: Queryable>() -> Vec<T> {
    let name = T::object_name();
    let mut res = vec![];

    for obj in raw_api::query(&format!("SELECT * FROM Win32_{name}")) {
        res.push(T::from(obj))
    }

    res
}
```

And once the user implements[^2] this new trait for `Fan`:

[^2]: Our new trait is essentially [`From<Object>`](https://doc.rust-lang.org/std/convert/trait.From.html) with the small addition of returning the name of the object to query, so a more idiomatic definition would be `trait Queryable: From<Object> { .. }` which requires two separate implementations.

```rust
impl Queryable for Fan {
    fn object_name() -> &'static str {
        "Fan"
    }

    fn from(obj: Object) -> Self {
        let name = if let Value::String(name) = obj.get_attr("Name") {
            name
        } else {
            panic!()
        };

        // .. repeat for the other fields ..

        Fan {
            name,
            active_cooling,
            desired_speed,
        }
    }
}
```

Then they can use our improved `query` function by specifying the return type:

```rust
let res: Vec<Fan> = api::query();
```

Which is a substantial improvement over the `raw_api:query` version we had before.

While simpler to do from the library's point of view, this approach forces the user 
to manually implement the `Queryable` trait for each and every new type they want to use,
which is verbose and error-prone and not really ergonomic, even without proper error handling or support for nested objects.

We could use dtolnay's [guide and learn to write procedural macros](https://github.com/dtolnay/proc-macro-workshop), create a macro that generates this implementation automatically[^3] and expose it to users,
but... we can use dtolnay's [`serde`](https://github.com/serde-rs/serde) which essentially does all of this already!

[^3]: And we didn't mention casing conversions (`ActiveCooling` vs `active_cooling`), struct name customizations (What if the user want the struct to be `struct SystemFan {}`?), safe integer conversions (`f1: u64` should - or maybe should not - accept a `u8` value?), `enum` support, and so much more!

## Serde to the Rescue

Serde is a _framework_ for serializing and deserializing data in Rust, 
which means that it defines traits (`Serialize` and `Deserialize`), as well as the ability to use `derive` to generate implementations for these traits at compile time.

It is this ability to to generate a `Deserialize` implementation and than use it to create instances of a type that's going to be useful to us.

I should preface this by saying that while we'll learn a lot about the internals of Serde 
(which, fair warning, is a complex topic even when focusing on a specific part like we'll do),
I'm not trying to claim that this is the most _conventional_ use case for Serde (see the [Alternatives](#alternatives) section for more about this). 
I do think the result is pretty neat though.

To use Serde, we usually need an additional library that leverages these traits to interact with different data formats.

For example, using `serde_json` looks like this for `Fan`:

```rust
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Fan {
    name: String,
    active_cooling: bool,
    desired_speed: u64,
}

let fan: Fan = serde_json::from_str(r#"{ "Name": "CPU1", "ActiveCooling": true, "DesiredSpeed": 1100 }"#)?;

println!("Fan `{}` is running at {} RPM", fan.name, fan.desired_speed);
```

_Note: Already, using Serde begins to pay off: `rename_all = "PascalCase"` is just what we need to keep our field names in `snake_case`._

Since Serde is able to create the user's struct (`Fan`) from different "data formats" (JSON, YAML, Postcard, ...),
_we_ can pretend to be another data format, and hitch a ride on the compile-time generated `Deserialize` impl provided by `derive(Deserialize)`.

If we can do that, we'll be able to create a `query` function that accepts any struct that implements `Deserialize`,
which would be convenient for our users: not only can they simply add `derive(Deserialize)` to their types, 
they also can use different Serde configurations (like the `rename_all` option) and the rich ecosystem around Serde (like the `serde_with` crate).

The bottom line is that we want to switch from our custom `Queryable` trait to Serde's `Deserialize` trait,
which would look a bit like this:

```diff
-fn query<T: Queryable>() -> Vec<T> { .. }
+fn query<T: Deserialize>() -> Vec<T> { .. }
```

Now, this is _much_ easier said than done, so this might be a good time to get a fresh pot of tea or coffee: 
we have a long road ahead of us (with lots and _lots_ of traits).

_Note: If you're less familiar with Rust, hold on to your whitespaces: this is where the deeper part of the deep dive starts, but feel free to go straight to the [summary](#summary)._

### Peeking Under the Hood

We know that we want to use the `Deserialize` trait in our `query` function,
but because it's almost never implemented or used directly, we need to learn more about it before we can do that.

So, to get a better sense of how the code above works, let's try to replace the `derive(Deserialize)` 
with a manual (and not feature complete)[^5] implementation of `Deserialize` for the `Fan` struct.

[^5]: You can use `cargo expand` to see what `derive(Deserialize)` does, but it's too verbose to really be educational. You can view it [here in the playground](https://play.rust-lang.org/?version=nightly&mode=debug&edition=2021&gist=46f52396029d4f805eb3570860e8ec4a).

We'll do this in a multiple stages, but it boils down to (1) defining new utility `struct`s and then (2) implementing some traits for them.

We'll start by looking at the definition of the `Deserialize` trait, and add an `impl` of it for `Fan`:

```rust
// The `Deserialize` trait:
pub trait Deserialize<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
       where D: Deserializer<'de>;
}

// Our implementation:
impl<'de> serde::Deserialize<'de> for Fan { .. }
```

You can safely ignore the [`'de` lifetime](https://serde.rs/lifetimes.html) for our use case (because we are not going to keep any data borrowed from the `Deserializer`),
and read the implementation definition as:

> For _any_ lifetime `'de`, this is the implementation of the `serde::Deserialize` trait (with that lifetime) for `Fan`.[^6]

The only method we need to implement is `fn deserialize`, which accepts a `Deserializer`. \
Again, **you can safely ignore the `'de` lifetime**, but you can read `D: serde::Deserializer<'de>` as:

> The generic type `D` implements the `serde::Deserializer` trait, and any _borrowed_ data it may produce will have the `'de` lifetime.

```rust
fn deserialize<D>(
    deserializer: D,
) ->  Result<Fan, D::Error>
where
    D: serde::Deserializer<'de> { .. }
```

[^6]: We can also say that the `'de` lifetime is an _unconstrained_ lifetime parameter. Contrast this with Serde's [`impl<'de: 'a, 'a> Deserialize<'de> for &'a str`](https://docs.rs/serde/latest/src/serde/de/impls.rs.html#741-748), which can be read as "For _any lifetime `'de` that's longer than `'a`_, ..", and [`BorrowedStrDeserializer`](https://docs.rs/serde/latest/serde/de/value/struct.BorrowedStrDeserializer.html), which holds a `&'de str` and only `impl<'de> de::Deserializer<'de> for BorrowedStrDeserializer<'de>`.

Unexpectedly, most of the interesting code doesn't live in the `Deserialize` impl at all[^4].

[^4]: Ok, _technically_, in the `derive(Deserialize)` impl, both `Visitor` and `impl de::Visitor` are defined **inside** the `deserialize` method, but I think the point stands.

```rust
impl<'de> serde::Deserialize<'de> for Fan {
    fn deserialize<D>(
        deserializer: D,
    ) ->  Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &'static [&'static str] = &[
            "Name",
            "ActiveCooling",
            "DesiredSpeed",
        ];
        
        deserializer.deserialize_struct(
            "Fan",
            FIELDS,
            // New type!
            FanVisitor {},
        )
    }
}
```

Let's break this down a little.

The `deserializer` argument (which can be, for example, `serde_json::Deserializer`) 
is the part that knows how to access the data we need from the "data format" (the JSON-formatted string, in this case).

The [`Deserializer`](https://docs.rs/serde/latest/serde/trait.Deserializer.html) trait has a cosy 32 methods,
but we are using just one: `deserialize_struct`.

```rust
trait Deserializer<'de> {
    type Error: Error;

    fn deserialize_struct<V>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
       where V: Visitor<'de>;
}
```

The most interesting thing about this function is the `Visitor` argument.

Consider the point-of-view of the `Deserializer`, as it processes `{ .., "DesiredSpeed": 1100 }`.
What is the problem that is actually being solved here?

the `FanVisitor` is constructing a `Fan`, but it could be building a `HashMap<String, i32>`, or any number of different things.

All the deserializer knows is that it's currently looking at a mapping, and that the current key is a string, while the value is some number.

It needs a way to tell us this, which is where the `Visitor` comes in:
by defining our own `FanVisitor`, we'll let the deserializer "drive" it by calling different methods on it for the data it sees, and `FanVisitor` will get to decide how to handle that data.

So our job is to give the deserializer a visitor it can use to feed data _back to us_.
This is a bit confusing at first, but seeing it in action will help clear things up.

Technically, we can't know _exactly_ how the `deserialize_struct` function is going to be implemented by different deserializers for different inputs,
but the `serde_json` deserializer [calls](https://docs.rs/serde_json/latest/src/serde_json/de.rs.html#1818-1864) `visitor.visit_map` for the example input above, so this is what we need to implement.

```rust
// Remember: `derive(Deserialize)` automatically generates this code 
// (both new struct and the impl) at compile time, based on the `Fan` struct definition.

// A struct with no fields, only needed so we can attach the impl to something.
struct FanVisitor;
impl<'de> serde::de::Visitor<'de> for FanVisitor {
    
    // The visitor specify what type it is going to produce 
    // (as indicated by the return type of `visit_map`).
    type Value = Fan;
    
    // The `map` is where the data from the deserializer is coming from.
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let (mut name, mut active_cooling, mut desired_speed) = (None, None, None);

        // Use the `map` we got from the deserializer.
        // `key`'s type is `&str`, which means we call `map.next_key::<&str>()`.
        while let Ok(Some(key)) = map.next_key() {
            // But, `next_value()` returns different types for different fields.
            // We'll explain how later, when we implement our own `Deserializer` and `MapAccess`.
            match key {
                "Name" => {
                    let val: String = map.next_value()?;
                    name = Some(val);
                }
                "ActiveCooling" => {
                    let val: bool = map.next_value()?;
                    active_cooling = Some(val);
                }
                // ..
            }
        }

        Ok(Fan { .. })
    }
}
```

We'll dive deeper into the `MapAccess` trait in the next section, but notice for now that we use it to get both keys and values, 
and that we (the `Visitor`) determine what types they should be (or rather, what type they should be _deserialized_ into).

There are various other `visit_{u64,string,bool,...}` functions, but by default they simply return an error, so we can implement only `visit_map` for this minimal example.

Having done this, the same code from before can now work without the `derive(Deserialize)`:

```rust
pub struct Fan { .. }
impl<'de> serde::Deserialize<'de> for Fan { .. }

let fan: Fan = serde_json::from_str(r#"{ "Name": "CPU1", "ActiveCooling": true, "DesiredSpeed": 1100 }"#)?;

println!("Manual Serde impl: Fan `{}` is running at {} RPM", fan.name, fan.desired_speed);
```

So, let's recap what we know about the trait-zoo of Serde:

1. `Deserialize::deserialize` accepts a `Deserializer`, and calls (for a `struct` like `Fan`) the `deserializer.deserialize_struct` function,
with a `Visitor` (like `FanVisitor`), which handles the creation of the `struct`.

2. The `Deserializer` (in this case `serde_json::Deserializer`) calls a `visitor.visit_` function (in this case, `visit_map`),
which provides data from the `Deserializer` (in this case, a `map` which implements `MapAccess`).

3. The `Visitor` now calls the `map`'s `next_{key,value}` functions which return the data needed to build the `struct`.

4. Once there is no `next_key` in the `map`, the `Visitor` finishes and returns the created value (in this case, an instance of `Fan`).
 
Or, more visually:

<style>
.serde-code-visualization pre code {
  font-size: 0.72em
}
</style>

```rust {class="serde-code-visualization"}
let fan = Fan::deserialize(serde_json::Deserializer::from_str(r#"{ "Name": "CPU1", "Active.."#));

impl Deserialize for Fan                            impl Deserializer for serde_json::Deserializer
┌ fn deserialize(deser)                             │
│   let visitor = FanVisitor {}                     │
│   deser.deserialize_struct(.., visitor) ─calls──► ├ fn deserialize_struct(.., visitor)
│                                                   │   let map = serde_json::de::MapAccess::new(..)
│   impl Visitor for FanVisitor                     │   
│   ┌ fn visit_map(map) ◄────────calls───────────────── return visitor.visit_map(map)
│   │   loop {                                          impl MapAccess for serde_json::de::MapAccess
│   │     key = map.next_key() ──────calls────────────► ┌ fn next_key()   // { ..,▼"Name": ..
│   │     /* since key is "Name" */                     │
│   │     name: String = map.next_value() ───calls────► └ fn next_value() // {         ..:▼"CPU1", ..
│   │   }
◄────── return Fan { ... }
```

Well, "more" being a key word here.

<img 
  src="/2025-05-serde-reflect/charlie_conspiracy.webp" 
  alt="Charlie Conspiracy (always Sunny In Philadelphia)" 
  title="Let's talk about the traits. Can we talk about the traits? I'm dying to talk about the traits." 
  width="60%" 
  loading="lazy"
/>

Now that we have a basic understanding of the flow let's leave the manual `Deserialize` impl and it's `FanVisitor` behind us,
and focus on our actual goal: a better `query` API.

```diff
-pub struct Fan { .. }
-impl<'de> serde::Deserialize<'de> for Fan { .. }
+#[derive(Debug, Deserialize)]
+#[serde(rename_all = "PascalCase")]
+pub struct Fan { .. }
```

## Implementing a Deserializer

The reason we opt to use `Deserialize` was _because_ Serde can `derive` it for the user's type. 
So our job is to implement the `query` function in a way that:

1. Accepts `T` that implements `Deserialize`.
2. Create `T`s from `raw_api::Object`s by calling _their_ `Deserialize::deserialize` function.

Since we now know much more about what the `Deserialize::deserialize` function does, 
let's go ahead and start by updating `query` to use `Deserialize`, with a small twist:

```rust
pub fn query<T: DeserializeOwned>() -> Vec<T> {
    // We'll fix this one later.
    let name = "Fan";
    let mut res = vec![];

    for obj in raw_api::query(&format!("SELECT * FROM Win32_{name}")) {
        // New type!
        let deser = ObjectDeserializer { obj };
        res.push(T::deserialize(deser).unwrap())
    }

    res
}
```

Instead of using `Deserialize<'de>`, we use [`DeserializeOwned`](https://docs.rs/serde/latest/serde/de/trait.DeserializeOwned.html), 
which is similar but forbids users from using types that borrow data (like `struct BorrowedFan<'a> { name: &'a str }`).[^7]

[^7]: Compare and contrast to [`serde_json::from_str`](https://docs.rs/serde_json/latest/serde_json/fn.from_str.html) and [`serde_json::from_reader`](https://docs.rs/serde_json/latest/serde_json/fn.from_reader.html).

The main difference between this and our [`Queryable` trait version](#crawl-before-you-can-walk) is 
that Serde introduces a new intermediary `struct` and trait (`T::deserialize(ObjectDeserializer { obj })` vs `T::from(obj)`):
the added level of indirection is what allows us to decouple the `Deserializer` from the user's `T`.

Serde's docs describe [implementing a deserializer](https://serde.rs/impl-deserializer.html) like so:

> The deserializer is responsible for mapping the input data into Serde's data model by invoking exactly one of the methods on the `Visitor` that it receives. \
> The `Deserializer` methods are called by a `Deserialize` impl as a hint to indicate what Serde data model type the `Deserialize` type expects to see in the input.

Similar to our [custom implementation](#peeking-under-the-hood), the `derive(Deserialize)` version is also going to call `deserialize_struct`, so we need to implement that:

```rust
struct ObjectDeserializer {
    obj: raw_api::Object,
}

impl<'de> Deserializer<'de> for ObjectDeserializer {
    // .. snip ..
    // There are all sorts of `deserialize_{bool,str,enum,...}`, 
    // but we can ignore them for now since we only want to support `struct`s.

    // Our implementation, which will be called by `T::Deserialize`:
    fn deserialize_struct<V>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }
}
```

Given all that we've seen so far, our `deserialize_struct` method will need to:

1. Define a new struct that implements [`serde::de::MapAccess`](https://docs.rs/serde/latest/serde/de/trait.MapAccess.html).
2. Call `visitor.visit_map` with that struct, which will need return the right `key`s and `value`s every time `next_key` and `next_value` are called.

Implementing the `MapAccess` trait requires only two functions: `next_key_seed` and `next_value_seed`.
They are very similar to `next_{key,value}` we called before, but with some additional flexibility provided by [`DeserializeSeed`](https://docs.rs/serde/latest/serde/de/trait.DeserializeSeed.html), which I'll get to in a bit.

```rust
trait MapAccess<'de> {
    type Error: Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
       where K: DeserializeSeed<'de>;
    
    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
       where V: DeserializeSeed<'de>;
       
    // ..
}
```

So we need to return each of the fields as the key in `next_key_seed`, and then return the field's value in the following call to `next_value_seed`.

We can define a new struct called `ObjectMapAccess` and use it like this:

```rust
struct ObjectMapAccess {
    // Nothing fancy, this is the type that we get 
    // by calling `fields.iter().peekable()`.
    fields: Peekable<Iter<'static, &'static str>>,
    obj: raw_api::Object,
}

// in deserialize_struct's body:
let map = ObjectMapAccess {
    fields: fields.iter().peekable(),
    obj: self.obj,
};

visitor.visit_map(map)
```

The basic logic in the implementation of `MapAccess` for `ObjectMapAccess` is to `peek()` the next field from `fields` in `next_key_seed`, 
then get it again via `next()` in the `next_value_seed` and call `obj.get_attr` on it.

But if we do this and look at the signatures of both functions, we'll see we aren't done yet:

```rust
impl<'de> serde::de::MapAccess<'de> for ObjectMapAccess {
    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: serde::de::DeserializeSeed<'de>,
    {
        if let Some(field) = self.fields.peek() {
            // Hmm.
        } else {
            Ok(None)
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::DeserializeSeed<'de>,
    {
        let current_field = self.fields.next().unwrap();

        let field_value = self.obj.get_attr(current_field);

        // Hmm.
    }
}
```

### Bad Seeds

The problem is that if we look at the return types for both functions, we need to return `K::Value` and `V::Value`, 
but we only know that `K` and `V` implement [`DeserializeSeed`](https://docs.rs/serde/latest/serde/de/trait.DeserializeSeed.html),
which looks like this in comparison to `Deserialize`:

```rust
trait Deserialize<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
       where D: Deserializer<'de>;
}

trait DeserializeSeed<'de> {
    // New.
    type Value;

    // Return type uses `Value`.
    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
       where D: Deserializer<'de>;
}
```

Don't worry about it too much, but roughly speaking it means that a `DeserializeSeed` value can have some state (vs. the regular `Deserialize` which cannot).

When a visitor calls `let name: String = map.next_value().unwrap()`, for example,
the type substitution results in:

```rust
fn next_value_seed(
    &mut self,
    seed: PhantomData<String>,
) -> Result<String, Self::Error> { ... }
```

and calling `seed.deserialize(..)` is [equivalent](https://docs.rs/serde/latest/serde/de/trait.DeserializeSeed.html#foreign-impls) to calling `String::deserialize(..)`.

_Note: [`PhantomData`](https://doc.rust-lang.org/std/marker/struct.PhantomData.html) is Rust-speak for "I care about the type of something, but don't have actual value of that type"._

With that out of the way, we can now _finally_ complete the trait puzzle!

### It's `Deserializer`s All the Way Down

The only thing we can do is to call `seed.deserialize` on something that is a `Deserializer`, since it's the only function provided by the `Deserialize/DeserializeSeed` traits.

For the field name, this turns out to be easy: we know the field name we have is a `&str`, 
and Serde provides a [`StrDeserializer`](https://docs.rs/serde/latest/serde/de/value/struct.StrDeserializer.html) which (as the name suggests) 
[implements](https://docs.rs/serde/latest/src/serde/de/value.rs.html#472-504) the `Deserializer` trait, and can be created from a `&str`.

Using the same visualization from before, if the _calling_ `Visitor` expects a `String` key (for example in `HashMap<String, _>`), we'll get something like this:

```rust {class="serde-code-visualization"}
// Same as `next_key_seed` after type substitution for `String`:
// `StrDeserializer` is defined as `struct StrDeserializer { value: &str, .. }`.
fn next_key() -> Result<String, _> { String::deserialize(StrDeserializer::new("Name")) }
let key: String = next_key();

impl Deserialize for String                        impl Deserializer for StrDeserializer
┌ fn deserialize(deser)                            │
│   let visitor = StringVisitor {}                 │
│   deser.deserialize_string(.., visitor) ─calls─► │ fn deserialize_string(.., visitor)
│                                                      let value = self.value
│   impl Visitor for StringVisitor
│   ┌ fn visit_str(value: &str) ◄────────calls──────── return visitor.visit_str(value)
◄────── return Ok(value.to_owned())
```

You can also think about this as the "base case" for the `Visitor`: somewhere down the stack, we'll want to construct some [primitive type](https://serde.rs/data-model.html) which has a `Deserialize` implementation that doesn't recurse any further.

This might seem a bit over the top in this case, but remember that `visit_map` needs to support any key type that can be deserialized (like when building a `HashMap<i32, i32>` from JSON - [playground](https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=2ca6ff42acab1e15fa2f39cf4326e3ba)).

Putting all of this together, we arrive at this implementation of `next_key_seed`:

```rust
if let Some(field) = self.fields.peek() {
    // Create a `StrDeserializer` for the &str.
    let field_deser = StrDeserializer::new(field);

    // Let the seed use the deserializer's data to create whatever type it needs.
    // `StrDeserializer` will call `visitor.visit_str(field)`
    // on whatever visitor is defined by the seed's impl.
    // For structs, the `derive` generates an `enum Fields { .. }`,
    // which will pick the right variant based on the provided &str.
    seed.deserialize(field_deser).map(Some)
} else {
    Ok(None)
}
```

Now, as we turn to `next_value_seed`, the solution is obvious: we need _another_ struct, and we need that struct to implement the `Deserializer` trait for `raw_api::Value`:

```rust
struct ValueDeserializer { value: raw_api::Value }

let current_field = self.fields.next().unwrap();
let field_value = self.obj.get_attr(current_field);

seed.deserialize(ValueDeserializer { value: field_value })
```

Before, when we implemented `Deserializer` for `ObjectDeserializer`, we didn't have what Serde calls "a self-describing data format":
we needed the hint from `deserialize_struct`'s `fields` parameter to know what data to get using the `get_attr` function.

This time, `raw_api::Value` _is_ actually a self-describing data format: it knows exactly which function of the visitor to call:

```rust
impl<'de> Deserializer<'de> for ValueDeserializer {
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de> {
        match self.value {
            raw_api::Value::Bool(b) => visitor.visit_bool(b),
            raw_api::Value::I1(v) => visitor.visit_i8(v),
            // ..
            raw_api::Value::UI8(v) => visitor.visit_u64(v),
            raw_api::Value::String(s) => visitor.visit_string(s),
            _ => todo!(),
        }
    }

    // Because this is common, Serde provides this macro that makes other
    // `deserialize_` functions use the `deserialize_any` we implemented.
    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum struct identifier ignored_any
    }
}
```

Crossing our fingers, we can update the `query` call:

```rust
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Fan { .. }

let res: Vec<Fan> = query();

for fan in res {
    if fan.active_cooling {
        println!(
            "Fan `{}` is running at {} RPM",
            fan.name, fan.desired_speed
        );
    }
}
```

Recompile and...

```bash
$ cargo run
Fan `CPU1` is running at 150 RPM
```

everything compiles and works as expected! Huzza!

## Recap

To recap, the final flow is:

1. User calls query with a `T` that implements `Deserialize`.
2. Query fetches the objects, and calls `T::Deserialize` with our `ObjectDeserializer`.
3. `T::Deserialize` calls `ObjectDeserializer::deserialize_struct` with `T`'s unique compile-time generated `Visitor`, like the `FanVisitor` we implemented.
4. `ObjectDeserializer::deserialize_struct` creates an `ObjectMapAccess`, and passes it to `T`'s `Visitor`.
5. `T`'s `Visitor` calls the map's `next_key` and `next_value` to build a new instance of `T`.
6. `next_key` returns a field name from the peekable iterator, followed by `next_value` which calls `get_attr` to get the `Value` of that field.
7. We create a new `ValueDeserializer` with the field's `Value`, and passes it to the seed's `Deserialize` (which is the `Deserialize` impl of `String`, then `bool`, and finally of `u64`).
8. `String`'s `Deserialize` calls `ValueDeserializer::deserialize_string`, which is forwarded to `deserialize_any`, which calls `visitor.visit_string(s)`, which is handled by `String`'s `Deserialize`'s `Visitor`.
9. `String`'s `Deserialize`'s `Visitor` only needs to return the `s` it got from that call.
10. Finally, after doing this for each field, `T`'s `Visitor` is done and returns a new instance of `T`.

You can also check the full code, which is less than 150 lines, in the [`v2_api.rs` file on GitHub](https://github.com/ohadravid/serde-reflect/blob/main/src/v2_api.rs).

As a bonus point, consider that this is pretty much a zero overhead abstraction: we are doing almost exactly what the manual `Queryable` implementation did,
with a few more indirections that can be optimized away by the compiler.

## Summary

So what did we learn today?

We saw how to use Rust's trait system to build interesting and ergonomic APIs, 
and we explored Serde's internals and its unique set of traits and used them 
to overcome some of the difficulties we had when using only custom traits to build our API.

We saw that while there are a lot of things going on behind the scenes, 
Serde doesn't rely on magic - it's built on a creative and sophisticated use of Rust's powerful trait system.

Before paying off one final debt and showing how [we can get the name of `T`](#going-a-step-further), 
let's wrap up by talking about some alternatives and tradeoffs.

### Alternatives

This is far from the only way to do this: building a procedural macro is perhaps the more "classic" solution to this problem,
and while for me the tradeoffs between maintainability, feature support, and ease of use still justify the use/abuse of Serde,
I still think there's room for a middle ground (perhaps like dtolnay's [reflect](https://docs.rs/reflect/latest/reflect/)).

On the other end of the spectrum, it's also possible to lean into code generation, 
like the [tonic](https://github.com/hyperium/tonic) and [prost](https://github.com/tokio-rs/prost) crates do for dealing with ProtoBufs and gRPC clients and servers. 
WMI also supports an RPC-like interface for method calling, and a first-class support for it would probably require something similar.
Again, it's all about the tradeoffs: focusing on data queries makes it harder to justify the added complexity 
(both in implementation for the library and in usage for users).

There are also different ORM-oriented crates that might be a good fit for this use case
(especially since we could also benefit from generating the SQL-like queries from typed objects),
but at the time the ecosystem was a lot less developed, and also much less stable: 
in more than 7 years, Serde's API didn't break once (still on version `1`), while the ORM ecosystem changed a lot since.

I think it would also be interesting to see how a different approach to compile-time code generation, like Zig's [comptime](https://zig.guide/language-basics/comptime/),
can support something similar.

Discuss on [r/rust](https://www.reddit.com/r/rust/comments/1kgwvwc/a_rust_api_inspired_by_python_powered_by_serde/), [lobsters](https://lobste.rs/s/8jjrva/rust_api_inspired_by_python_powered_by), [HN](https://news.ycombinator.com/item?id=43954858)! 👋

## Future Work

You can explore the entire source code for this article in [this GitHub repo](https://github.com/ohadravid/serde-reflect),
or check the code of the [`wmi-rs` crate], which also has:

1. Nested objects support, like `struct Win32_DiskDriveToDiskPartition { Antecedent: Win32_DiskDrive, Dependent: Win32_DiskPartition }`.
2. New-type support and HashMap support, like `struct Fan(HashMap<String, Value>)`.
3. Enum support, like `enum CIM_CoolingDevice { Win32_Fan(Fan), Win32_HeatPipe(HeatPipe), .. }`.
4. Deserialization of `chrono` and `time` objects from strings (using `struct WMIDateTime(pub DateTime<FixedOffset>)` with a custom deserializer).
5. Serialization support (like creating an `Object` from `Fan`).

(and, obviously, an implementation of the "raw" API for using Windows' WMI infrastructure).

## Going a Step Further

Remember this:

```rust
pub fn query<'de, T>() -> Vec<T>
where
    T: Deserialize<'de>,
{
    // We'll fix this one later.
    let name = "Fan";
    
    // ..
}
```

Well, we saw that the `deserialize_struct` function we implemented accept both the `fields` argument (which we used), and also the `name` argument.

But... we can't use that since we don't have an `Object` to deserialize yet.

The answer? You guessed it! _Yet another struct and a `Deserializer` impl_, this time taken directly from Serde [issue #1110](https://github.com/serde-rs/serde/issues/1110):

```rust
// A new `Deserializer` that can write the `struct`'s name
// to a caller's variable.
struct StructNameDeserializer<'a> {
    name: &'a mut Option<&'static str>,
}

impl<'de, 'a> Deserializer<'de> for StructNameDeserializer<'a> {
    fn deserialize_struct<V>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        // Store the name.
        *self.name = Some(name);
        self.deserialize_any(visitor)
    }

    // Always end with an error, since we don't want to deserialize anything.
    fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        Err(de::Error::custom("I'm just here for the metadata"))
    }

    // .. snip ..
}

let mut name = None;

let _ = T::deserialize(StructNameDeserializer {
    name: &mut name,
});
```

Since `T`'s `Deserialize` impl will call `deserialize_struct` with the `struct`'s name,
our deserializer will capture that name and write it back to the given `name` variable,
then end with an error (that is ignored by the caller).

One neat feature we get for free here is that using `#[serde(rename = "..")]` on the `struct` works as you might expect:

```rust
#[derive(Default, Deserialize)]
#[serde(rename = "Fan")]
#[serde(rename_all = "PascalCase")]
pub struct OsFan {
    name: String,
    active_cooling: bool,
    desired_speed: u64,
}

assert_eq!(struct_name::<OsFan>(), "Fan");
```


_Special thanks to [Yoav](https://github.com/yoavrv) and [Ido](https://github.com/idobenamram) for reviewing earlier drafts of this article._