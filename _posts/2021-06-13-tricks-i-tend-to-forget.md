---
title: "Tricks I Tend To Forget While Programming"
date: 2021-06-13
categories: [programming]
tags: [programming]     # TAG names should always be lowercase
---

## Introduction

This post is a personal reference for mathematical and programming tricks that I find useful but tend to forget over time. Rather than re-deriving them from scratch each time, I collect them here with short proofs so I can quickly recall both the result and why it works.

<link rel="icon" type="image/png" href="../../assets/img/favicons/favicon-96x96.png" sizes="96x96" />
<link rel="icon" type="image/svg+xml" href="../../assets/img/favicons/favicon.svg" />
<link rel="shortcut icon" href="../../assets/img/favicons/favicon.ico" />
<link rel="apple-touch-icon" sizes="180x180" href="../../assets/img/favicons/apple-touch-icon.png" />

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Sum of the first n integers

The sum of the first $$n$$ integers satisfies:

$$\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$$

**Proof (Gauss's pairing trick).** Let $$S = 1 + 2 + \cdots + n$$. Write the sum twice, once forwards and once backwards, then add term by term:

$$S = n \;+\; (n-1) \;+\; \cdots \;+\; 2 \;+\; 1$$

$$S = n + 1 \;+\; (n - 1 + 2) \;+\; (n - 2 + 3) + \cdots \;+\; (2 + n - 1) \;+\; 1 + n$$

$$2S = \underbrace{(n+1) + (n+1) + \cdots + (n+1)}_{n \text{ terms}} = n(n+1)$$

Dividing both sides by 2:

$$S = \frac{n(n+1)}{2}$$
