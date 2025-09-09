---
title: "How to Build Python Code with Bazel (and Why)"
summary: ""
date: 2025-09-09T06:00:00+00:00
tags: ["python", "tooling", "build", "bazel"]
type: post
showTableOfContents: false
image: "/2025-09-hello-bazel/bazel_graph.webp"
weight: 5
---

How can you build Python code with [Bazel](https://bazel.build/)? Why would you even want to do that?

This is the topic of a short talk I gave at PyCon IL this year.
Below are the slides, starting with _why_ this is a problem that you might have.
You can also jump right into [the part about how Bazel can solve that problem](#how-bazel-solves-these-problems),
or [check the GitHub repo](https://github.com/ohadravid/hello-bazel-pycon-il).

Happy building!

-----

Hi! My name is Ohad, and I care (_maybe a little too much_) about things being as fast as possible.

Today I want to show you how to build Python code with Bazel - and, more importantly, why you’d even want to do that in the first place.

<img src="/2025-09-hello-bazel/hello_bazel_01.svg" width="75%" style="aspect-ratio: 16 / 9;" />

## The Problem

So let’s say you’ve got an app. A nice little app. Let’s say: Uber, but for dog walkers.

<img src="/2025-09-hello-bazel/hello_bazel_02.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

You’ve got a Python backend. \
It has a few dependencies. \
You build a Docker image and deploy it on K8s. \
And of course - there’s a CI pipeline running all the tests.

After all, we aren’t animals.


<img src="/2025-09-hello-bazel/hello_bazel_03.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>


One day, you’re tasked with implementing a new feature for the dog-walking app:
let users upload a picture of their dog.
Then - with a bit of AI - recognize the breed of the dog and save the user some typing!

Honestly, not too hard. And yes - it really takes like 10 lines of Python.

<img src="/2025-09-hello-bazel/hello_bazel_04.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

Of course, you add a test.

You open a PR.

And you’re just about to Alt-Tab over to the Wolt page - when suddenly…

<img src="/2025-09-hello-bazel/hello_bazel_05.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

The CI tests fail.
Okay, okay - probably just a missing dependency.
Should be an easy fix!

<img src="/2025-09-hello-bazel/hello_bazel_06.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

Which only takes… three hours to sort out.
Turns out the UV version in CI was too old.
Then the lockfile didn’t match, because you’re on macOS and CI is on Linux.
And… which version of Torch did we even want to install? So many choices!

But in the end, the tests pass. Victory!

<img src="/2025-09-hello-bazel/hello_bazel_07.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>


Buttttt, in production, the code crashes with a ton of "413 Request Too Large" errors - 
because the images are too big for Flask’s default settings.

Good thing you can do a rollback, huh?

<img src="/2025-09-hello-bazel/hello_bazel_08.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>


Suddenly, the CTO is in your Slack DMs.

"How did the tests fail catch this?" \
"How do we make sure it doesn’t happen again?"

And these are good questions!

<img src="/2025-09-hello-bazel/hello_bazel_09.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

So you do the right thing and add an integration test.

But now you have to test the production image in CI before every merge.
And it’s slow. \
And painful.

These tests also run every time you change something unrelated, \
and you find yourself waiting for the CI to finish even more than you wait for the Wolt delivery.

<img src="/2025-09-hello-bazel/hello_bazel_10.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

At this point, you start wondering - \
maybe you should open a microbrewery in a kibbutz and grow a mustache.

Maybe you already have a mustache!

Either way, coming to work isn’t that fun anymore.

<img src="/2025-09-hello-bazel/hello_bazel_11.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

## How Bazel Solves These Problems

These are the kinds of problems Bazel can solve:

It knows about our external dependencies (like Torch), \
and the dependencies between different parts of the project itself (like between the tests and the server code).

That way, it can avoid unnecessary work every time we want to do something with our code.

<img src="/2025-09-hello-bazel/hello_bazel_12.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>


Basically, it looks like this: we have configuration files written in a language called Starlark, which is _almost_ Python.

We define different "targets" that we can ask Bazel to build.

Here, we define a target for a Python library:
we specify where the library’s sources are, how its imports should work, and which external packages it depends on - like Torch and Flask.

<img src="/2025-09-hello-bazel/hello_bazel_13.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

We have different types of targets, which can depend on each other.
For example, our main target depends on a library, and we indicate that using the library target’s name, prefixed with a colon.

And that’s basically how most of our Bazel files look: for instance,
our integration test depends on an Ubuntu-based image, which depends on layers that depend on main that depends on the library.

And why is all this useful?

<img src="/2025-09-hello-bazel/hello_bazel_14.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

For example, if I change the README, since it’s not part of the dependency graph for the integration test, when I try to run it, Bazel just tells me: \
"Don't worry about it buddy, I remember these tests already passed. Nothing changed, no need to run them now."

(Yeah, Bazel is a bit optimistic that way.)

<img src="/2025-09-hello-bazel/hello_bazel_15.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

And if I do change the code and want to build the image,
because the rules we’ve defined know how to distinguish between our code and external dependencies,
building the image stays very fast.

<img src="/2025-09-hello-bazel/hello_bazel_16.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

One last example: what if we want to add a native Rust extension?

We don’t need to build wheels or anything special. \
All that’s needed is an additional target specifying the Rust files we depend on, and...

<img src="/2025-09-hello-bazel/hello_bazel_17.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

We add it as a dependency to the library, and we’re done!

Bazel takes care of everything: it will download the Rust compiler, \
build the extension against the correct Python, \
handle the imports, \
and of course, it knows to redo that only when the Rust files change (and never when they don’t).

<img src="/2025-09-hello-bazel/hello_bazel_18.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

In general, Bazel gets more useful as your project grows in complexity.

There’s a non-trivial upfront setup cost, \
but the more services and libraries you have,
and especially if your project spans multiple languages or targets multiple platforms,
the more you gain from it.

And one last thing: Bazel is far from an "all-or-nothing" tool.
The more effort you put in, the faster and more reliable your builds become,
but even a relatively modest setup can deliver significant benefits.

<img src="/2025-09-hello-bazel/hello_bazel_19.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>

And that’s it! Now you know that you can build Python code with Bazel, and maybe it’ll come in handy for your next project.

Here’s a [link to a sample repo](https://github.com/ohadravid/hello-bazel-pycon-il) showing the full setup we have here:
building a Docker image, classifying a dog image with Torch, running an integration test, a native Rust extension, and more!

Thanks!

<img src="/2025-09-hello-bazel/hello_bazel_20.svg" width="75%" loading="lazy" style="aspect-ratio: 16 / 9;"/>
