# Mathematical Foundations of Special Functions

This document provides comprehensive mathematical foundations, proofs, and derivations for the special functions implemented in scirs2-special.

## Table of Contents

1. [Gamma and Beta Functions](#gamma-and-beta-functions)
2. [Bessel Functions](#bessel-functions)
3. [Error Functions](#error-functions)
4. [Orthogonal Polynomials](#orthogonal-polynomials)
5. [Hypergeometric Functions](#hypergeometric-functions)
6. [Wright Functions](#wright-functions)
7. [Elliptic Integrals](#elliptic-integrals)
8. [Spherical Harmonics](#spherical-harmonics)
9. [Mathieu Functions](#mathieu-functions)
10. [Asymptotic Analysis](#asymptotic-analysis)

---

## Gamma and Beta Functions

### Gamma Function Definition

The Gamma function is defined by the integral:

$$\Gamma(z) = \int_0^{\infty} t^{z-1} e^{-t} dt, \quad \Re(z) > 0$$

### Fundamental Properties

**Recurrence Relation:**
$$\Gamma(z+1) = z\Gamma(z)$$

**Proof:** Using integration by parts on the defining integral:
$$\Gamma(z+1) = \int_0^{\infty} t^z e^{-t} dt$$

Let $u = t^z$ and $dv = e^{-t}dt$. Then $du = zt^{z-1}dt$ and $v = -e^{-t}$.

$$\Gamma(z+1) = \left[-t^z e^{-t}\right]_0^{\infty} + z\int_0^{\infty} t^{z-1} e^{-t} dt$$

The boundary term vanishes, giving us $\Gamma(z+1) = z\Gamma(z)$. □

**Special Values:**
- $\Gamma(1) = 0! = 1$
- $\Gamma(n+1) = n!$ for $n \in \mathbb{N}_0$
- $\Gamma(1/2) = \sqrt{\pi}$

**Proof of $\Gamma(1/2) = \sqrt{\pi}$:**

$$\Gamma(1/2) = \int_0^{\infty} t^{-1/2} e^{-t} dt$$

Substituting $t = u^2$, $dt = 2u du$:

$$\Gamma(1/2) = \int_0^{\infty} (u^2)^{-1/2} e^{-u^2} \cdot 2u du = 2\int_0^{\infty} e^{-u^2} du$$

The Gaussian integral $\int_{-\infty}^{\infty} e^{-u^2} du = \sqrt{\pi}$, so $\int_0^{\infty} e^{-u^2} du = \sqrt{\pi}/2$.

Therefore, $\Gamma(1/2) = 2 \cdot \sqrt{\pi}/2 = \sqrt{\pi}$. □

### Stirling's Approximation

For large $|z|$:
$$\Gamma(z) \sim \sqrt{\frac{2\pi}{z}} \left(\frac{z}{e}\right)^z$$

**Derivation:** Using the saddle-point method on the integral representation:

$$\ln \Gamma(z) = \int_0^{\infty} [(z-1)\ln t - t] dt$$

The saddle point occurs at $t = z-1$, yielding the asymptotic expansion.

### Beta Function

The Beta function is defined as:
$$B(a,b) = \int_0^1 t^{a-1}(1-t)^{b-1} dt = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$$

**Proof of the Gamma function relation:**

Starting with the product:
$$\Gamma(a)\Gamma(b) = \int_0^{\infty} s^{a-1} e^{-s} ds \int_0^{\infty} t^{b-1} e^{-t} dt$$

Converting to polar coordinates $(s = r\cos^2\theta, t = r\sin^2\theta)$:
$$= 4\int_0^{\pi/2} \int_0^{\infty} r^{a+b-1} e^{-r} \cos^{2a-1}\theta \sin^{2b-1}\theta \, dr \, d\theta$$

$$= \Gamma(a+b) \cdot 2\int_0^{\pi/2} \cos^{2a-1}\theta \sin^{2b-1}\theta \, d\theta$$

Using the substitution $u = \cos^2\theta$ yields $B(a,b)$. □

---

## Bessel Functions

### Bessel Differential Equation

The Bessel differential equation is:
$$z^2 \frac{d^2y}{dz^2} + z\frac{dy}{dz} + (z^2 - \nu^2)y = 0$$

### Series Solutions

**Bessel Functions of the First Kind:**
$$J_\nu(z) = \left(\frac{z}{2}\right)^\nu \sum_{k=0}^{\infty} \frac{(-1)^k}{k!\Gamma(\nu+k+1)} \left(\frac{z}{2}\right)^{2k}$$

**Derivation:** Using the Frobenius method, we assume a solution of the form:
$$y = z^r \sum_{n=0}^{\infty} a_n z^n$$

Substituting into the differential equation and equating coefficients yields the recurrence relations that lead to the series representation. □

**Orthogonality Relations:**
$$\int_0^1 x J_\mu(\alpha_{\mu,m} x) J_\mu(\alpha_{\mu,n} x) dx = \frac{\delta_{mn}}{2} [J_{\mu+1}(\alpha_{\mu,m})]^2$$

where $\alpha_{\mu,m}$ are the zeros of $J_\mu(x)$.

### Asymptotic Behavior

For large $|z|$:
$$J_\nu(z) \sim \sqrt{\frac{2}{\pi z}} \cos\left(z - \frac{\nu\pi}{2} - \frac{\pi}{4}\right)$$

**Derivation using Stationary Phase Method:**

The integral representation:
$$J_\nu(z) = \frac{1}{\pi} \int_0^\pi \cos(z\sin\theta - \nu\theta) d\theta$$

For large $z$, the main contribution comes from the stationary points where $\frac{d}{d\theta}(z\sin\theta - \nu\theta) = 0$, i.e., $z\cos\theta = \nu$.

For $\nu \ll z$, the stationary point is near $\theta = \pi/2$, leading to the asymptotic formula. □

### Modified Bessel Functions

**Modified Bessel Functions of the First Kind:**
$$I_\nu(z) = i^{-\nu} J_\nu(iz) = \left(\frac{z}{2}\right)^\nu \sum_{k=0}^{\infty} \frac{1}{k!\Gamma(\nu+k+1)} \left(\frac{z}{2}\right)^{2k}$$

**Asymptotic behavior for large $z$:**
$$I_\nu(z) \sim \frac{e^z}{\sqrt{2\pi z}} \left(1 - \frac{4\nu^2-1}{8z} + O(z^{-2})\right)$$

---

## Error Functions

### Definition and Integral Representation

The error function is defined as:
$$\operatorname{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2} dt$$

### Series Expansion

$$\operatorname{erf}(z) = \frac{2}{\sqrt{\pi}} \sum_{n=0}^{\infty} \frac{(-1)^n z^{2n+1}}{n!(2n+1)}$$

**Derivation:** From the Taylor series of $e^{-t^2}$:
$$e^{-t^2} = \sum_{n=0}^{\infty} \frac{(-t^2)^n}{n!} = \sum_{n=0}^{\infty} \frac{(-1)^n t^{2n}}{n!}$$

Integrating term by term:
$$\operatorname{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z \sum_{n=0}^{\infty} \frac{(-1)^n t^{2n}}{n!} dt = \frac{2}{\sqrt{\pi}} \sum_{n=0}^{\infty} \frac{(-1)^n z^{2n+1}}{n!(2n+1)}$$ □

### Asymptotic Expansion

For large $|z|$:
$$\operatorname{erfc}(z) \sim \frac{e^{-z^2}}{z\sqrt{\pi}} \sum_{n=0}^{\infty} \frac{(-1)^n (2n-1)!!}{(2z^2)^n}$$

**Derivation using Integration by Parts:**

Starting with:
$$\operatorname{erfc}(z) = \frac{2}{\sqrt{\pi}} \int_z^{\infty} e^{-t^2} dt$$

Let $u = e^{-t^2}$ and $dv = dt$. Then $du = -2te^{-t^2}dt$ and $v = t$.

Repeated integration by parts yields the asymptotic series. □

### Complex Error Function (Faddeeva Function)

$$w(z) = e^{-z^2} \operatorname{erfc}(-iz)$$

This function satisfies:
$$w'(z) = -2z w(z) + \frac{2i}{\sqrt{\pi}}$$

---

## Orthogonal Polynomials

### Legendre Polynomials

**Rodrigues' Formula:**
$$P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n}(x^2-1)^n$$

**Generating Function:**
$$\frac{1}{\sqrt{1-2xt+t^2}} = \sum_{n=0}^{\infty} P_n(x) t^n, \quad |t| < 1$$

**Orthogonality:**
$$\int_{-1}^1 P_m(x) P_n(x) dx = \frac{2}{2n+1} \delta_{mn}$$

**Proof of Orthogonality:** Using Rodrigues' formula and integration by parts:

$$\int_{-1}^1 P_m(x) P_n(x) dx = \frac{1}{2^{m+n} m! n!} \int_{-1}^1 \frac{d^m}{dx^m}(x^2-1)^m \frac{d^n}{dx^n}(x^2-1)^n dx$$

For $m \neq n$, repeated integration by parts shows this integral vanishes. □

### Hermite Polynomials

**Physicist's Hermite Polynomials:**
$$H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n} e^{-x^2}$$

**Generating Function:**
$$e^{2xt-t^2} = \sum_{n=0}^{\infty} H_n(x) \frac{t^n}{n!}$$

**Orthogonality:**
$$\int_{-\infty}^{\infty} H_m(x) H_n(x) e^{-x^2} dx = 2^n n! \sqrt{\pi} \delta_{mn}$$

### Laguerre Polynomials

**Generalized Laguerre Polynomials:**
$$L_n^{(\alpha)}(x) = \frac{x^{-\alpha} e^x}{n!} \frac{d^n}{dx^n}(e^{-x} x^{n+\alpha})$$

**Orthogonality:**
$$\int_0^{\infty} x^\alpha e^{-x} L_m^{(\alpha)}(x) L_n^{(\alpha)}(x) dx = \frac{\Gamma(n+\alpha+1)}{n!} \delta_{mn}$$

---

## Hypergeometric Functions

### Gaussian Hypergeometric Function

$$_2F_1(a,b;c;z) = \sum_{n=0}^{\infty} \frac{(a)_n (b)_n}{(c)_n} \frac{z^n}{n!}$$

where $(a)_n = a(a+1)\cdots(a+n-1)$ is the Pochhammer symbol.

### Integral Representation

For $\Re(c) > \Re(b) > 0$:
$$_2F_1(a,b;c;z) = \frac{\Gamma(c)}{\Gamma(b)\Gamma(c-b)} \int_0^1 t^{b-1}(1-t)^{c-b-1}(1-zt)^{-a} dt$$

### Kummer's Function

$$_1F_1(a;c;z) = \sum_{n=0}^{\infty} \frac{(a)_n}{(c)_n} \frac{z^n}{n!}$$

**Differential Equation:**
$$z\frac{d^2w}{dz^2} + (c-z)\frac{dw}{dz} - aw = 0$$

---

## Wright Functions

### Wright Bessel Function

The Wright Bessel function is defined by:
$$J_{\rho,\beta}(z) = \sum_{k=0}^{\infty} \frac{(-z)^k}{k! \Gamma(\rho k + \beta)}$$

### Asymptotic Behavior

For large $|z|$ and $\rho > 0$:
$$J_{\rho,\beta}(z) \sim \frac{1}{\sqrt{2\pi\rho}} z^{(\beta-1)/(2\rho)} \exp\left(\rho \left(\frac{z}{\rho}\right)^{1/\rho}\right)$$

**Derivation using Saddle Point Method:**

The integral representation:
$$J_{\rho,\beta}(z) = \frac{1}{2\pi i} \int_{\mathcal{C}} \Gamma(-s) \frac{z^s}{\Gamma(\beta - \rho s)} ds$$

For large $|z|$, the saddle point occurs at $s = s_0$ where the exponent is stationary. This leads to the asymptotic formula through standard saddle-point analysis. □

### Wright Omega Function

The Wright omega function $\omega(z)$ is defined as the solution to:
$$\omega e^\omega = z$$

It satisfies the functional equation:
$$\omega(z e^z) = z$$

---

## Elliptic Integrals

### Complete Elliptic Integrals

**Elliptic Integral of the First Kind:**
$$K(k) = \int_0^{\pi/2} \frac{d\theta}{\sqrt{1-k^2\sin^2\theta}}$$

**Series Expansion:**
$$K(k) = \frac{\pi}{2} \sum_{n=0}^{\infty} \left[\frac{(2n-1)!!}{(2n)!!}\right]^2 k^{2n}$$

**Elliptic Integral of the Second Kind:**
$$E(k) = \int_0^{\pi/2} \sqrt{1-k^2\sin^2\theta} \, d\theta$$

### Jacobi Elliptic Functions

The Jacobi elliptic functions $\operatorname{sn}(u,k)$, $\operatorname{cn}(u,k)$, and $\operatorname{dn}(u,k)$ are defined as the inverse of elliptic integrals.

**Addition Formulas:**
$$\operatorname{sn}(u+v,k) = \frac{\operatorname{sn}u \operatorname{cn}v \operatorname{dn}v + \operatorname{sn}v \operatorname{cn}u \operatorname{dn}u}{1 - k^2 \operatorname{sn}^2u \operatorname{sn}^2v}$$

---

## Spherical Harmonics

### Definition

$$Y_\ell^m(\theta,\phi) = \sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m)!}{(\ell+m)!}} P_\ell^m(\cos\theta) e^{im\phi}$$

where $P_\ell^m$ are the associated Legendre polynomials.

### Orthonormality

$$\int_0^{2\pi} \int_0^{\pi} Y_{\ell'}^{m'}(\theta,\phi)^* Y_\ell^m(\theta,\phi) \sin\theta \, d\theta \, d\phi = \delta_{\ell'\ell} \delta_{m'm}$$

### Addition Theorem

$$P_\ell(\cos\gamma) = \frac{4\pi}{2\ell+1} \sum_{m=-\ell}^{\ell} Y_\ell^m(\theta_1,\phi_1)^* Y_\ell^m(\theta_2,\phi_2)$$

where $\cos\gamma = \cos\theta_1\cos\theta_2 + \sin\theta_1\sin\theta_2\cos(\phi_1-\phi_2)$.

---

## Mathieu Functions

### Mathieu Differential Equation

$$\frac{d^2y}{dz^2} + (a - 2q\cos 2z)y = 0$$

The characteristic values $a = a_n(q)$ and $b_n(q)$ are determined by the requirement of periodic or semi-periodic solutions.

### Floquet Theory

Solutions have the form:
$$y(z) = e^{i\mu z} P(z)$$

where $P(z)$ is periodic with period $\pi$ or $2\pi$.

---

## Asymptotic Analysis

### Steepest Descent Method

For integrals of the form:
$$I(\lambda) = \int_{\mathcal{C}} g(z) e^{\lambda f(z)} dz$$

As $\lambda \to \infty$, the main contribution comes from saddle points where $f'(z) = 0$.

**Leading Asymptotic Term:**
$$I(\lambda) \sim g(z_0) e^{\lambda f(z_0)} \sqrt{\frac{2\pi}{\lambda |f''(z_0)|}}$$

### Watson's Lemma

If $f(t) \sim \sum_{n=0}^{\infty} a_n t^{\alpha_n}$ as $t \to 0^+$ with $0 \leq \alpha_0 < \alpha_1 < \cdots$, then:

$$\int_0^{\infty} f(t) e^{-\lambda t} dt \sim \sum_{n=0}^{\infty} a_n \frac{\Gamma(\alpha_n + 1)}{\lambda^{\alpha_n + 1}}$$

as $\lambda \to +\infty$.

---

## Convergence and Error Analysis

### Numerical Stability

**Condition Number:** For a function $f$, the relative condition number is:
$$\kappa = \left|\frac{xf'(x)}{f(x)}\right|$$

**Forward Error Analysis:** If $\tilde{f}(x)$ is a computed approximation to $f(x)$:
$$\frac{|\tilde{f}(x) - f(x)|}{|f(x)|} \approx \kappa \cdot \epsilon_{machine}$$

**Example: Gamma Function Conditioning**

For the gamma function, the condition number is:
$$\kappa_\Gamma(x) = \left|\frac{x\psi(x)}{\Gamma(x)} \cdot \Gamma(x)\right| = |x\psi(x)|$$

where $\psi(x) = \Gamma'(x)/\Gamma(x)$ is the digamma function.

Near integer values, $\psi(n) \approx \ln(n) - 1/n$, so $\kappa_\Gamma(n) \approx n|\ln(n) - 1/n|$.

This shows that the gamma function becomes increasingly ill-conditioned for large arguments.

**Stability Analysis for Bessel Functions:**

For Bessel functions $J_\nu(x)$, the condition number is:
$$\kappa_{J_\nu}(x) = \left|\frac{x J_\nu'(x)}{J_\nu(x)}\right|$$

Using the recurrence relation $J_\nu'(x) = \frac{\nu}{x}J_\nu(x) - J_{\nu+1}(x)$:
$$\kappa_{J_\nu}(x) = \left|\nu - \frac{x J_{\nu+1}(x)}{J_\nu(x)}\right|$$

Near zeros of $J_\nu(x)$, this can become very large, indicating potential numerical instability.

### Series Truncation Errors

For alternating series satisfying the conditions of the alternating series test, the error is bounded by the first neglected term:

$$\left|\sum_{n=0}^{N} (-1)^n a_n - \sum_{n=0}^{\infty} (-1)^n a_n\right| \leq a_{N+1}$$

**Advanced Error Bounds for Hypergeometric Series:**

For the hypergeometric series $_1F_1(a;c;z)$, when truncated at term $N$:

$$\left|_1F_1(a;c;z) - \sum_{n=0}^{N} \frac{(a)_n z^n}{(c)_n n!}\right| \leq \frac{|(a)_{N+1}||z|^{N+1}}{|(c)_{N+1}|(N+1)!} \cdot R_N(a,c,z)$$

where $R_N(a,c,z)$ is a remainder factor that depends on the convergence properties.

**Richardson Extrapolation for Asymptotic Series:**

For asymptotic series that diverge, optimal truncation occurs at the smallest term. For Stirling's series:

$$\ln\Gamma(z) = (z-\frac{1}{2})\ln z - z + \frac{1}{2}\ln(2\pi) + \sum_{k=1}^{n} \frac{B_{2k}}{2k(2k-1)z^{2k-1}} + R_n(z)$$

The optimal truncation point $n^*$ satisfies $|a_{n^*}| \leq |a_{n^*+1}|$, giving an error bound of approximately $|a_{n^*}|$.

### Computational Complexity Analysis

**Time Complexity:**
- Series evaluation (truncated): $O(N)$ where $N$ is the number of terms
- Continued fraction evaluation: $O(\log \epsilon^{-1})$ for accuracy $\epsilon$
- Asymptotic formula evaluation: $O(1)$

**Space Complexity:**
- Most implementations: $O(1)$ auxiliary space
- Precomputed tables: $O(N)$ where $N$ is table size
- Recursive algorithms: $O(\log N)$ stack space

**Algorithmic Stability:**

The choice of algorithm affects numerical stability:

1. **Forward vs. Backward Recurrence:** For Bessel functions with large order, backward recurrence is more stable
2. **Continued Fractions vs. Series:** Continued fractions often provide better stability for complex arguments
3. **Asymptotic vs. Series:** Asymptotic expansions may be more stable for large arguments despite being divergent

---

## Struve Functions

### Definition and Differential Equation

The Struve functions $H_\nu(x)$ and $L_\nu(x)$ are solutions to the non-homogeneous Bessel equation:
$$x^2 \frac{d^2y}{dx^2} + x\frac{dy}{dx} + (x^2 - \nu^2)y = \frac{4(\frac{x}{2})^{\nu+1}}{\sqrt{\pi}\Gamma(\nu + \frac{1}{2})}$$

### Series Representation

**Struve Function $H_\nu(x)$:**
$$H_\nu(x) = \left(\frac{x}{2}\right)^{\nu+1} \sum_{k=0}^{\infty} \frac{(-1)^k}{k!\Gamma(k+\nu+\frac{3}{2})} \left(\frac{x}{2}\right)^{2k}$$

**Modified Struve Function $L_\nu(x)$:**
$$L_\nu(x) = -i e^{-i\nu\pi/2} H_\nu(ix) = \left(\frac{x}{2}\right)^{\nu+1} \sum_{k=0}^{\infty} \frac{1}{k!\Gamma(k+\nu+\frac{3}{2})} \left(\frac{x}{2}\right)^{2k}$$

### Asymptotic Behavior

For large $|x|$:
$$H_\nu(x) \sim Y_\nu(x) + \frac{1}{\pi} \left(\frac{2}{x}\right)^{\nu+1} \Gamma\left(\nu + \frac{1}{2}\right)$$

**Proof:** The asymptotic expansion follows from the integral representation:
$$H_\nu(x) = \frac{2(\frac{x}{2})^\nu}{\sqrt{\pi}\Gamma(\nu+\frac{1}{2})} \int_0^1 \sin(x\sqrt{1-t^2}) (1-t^2)^{\nu-\frac{1}{2}} dt$$

For large $x$, the oscillatory integral can be evaluated using stationary phase methods. □

### Integral Representations

$$H_\nu(x) = \frac{2(\frac{x}{2})^\nu}{\sqrt{\pi}\Gamma(\nu+\frac{1}{2})} \int_0^1 \sin(x\sqrt{1-t^2}) (1-t^2)^{\nu-\frac{1}{2}} dt$$

$$L_\nu(x) = \frac{2(\frac{x}{2})^\nu}{\sqrt{\pi}\Gamma(\nu+\frac{1}{2})} \int_0^1 \sinh(x\sqrt{1-t^2}) (1-t^2)^{\nu-\frac{1}{2}} dt$$

---

## Parabolic Cylinder Functions

### Weber's Differential Equation

The parabolic cylinder functions are solutions to Weber's equation:
$$\frac{d^2y}{dx^2} + \left(\nu + \frac{1}{2} - \frac{x^2}{4}\right)y = 0$$

### Principal Solutions

**Weber Function $D_\nu(x)$:**
$$D_\nu(x) = 2^{\nu/2} e^{-x^2/4} \left[\frac{\sqrt{\pi}}{\Gamma(\frac{1-\nu}{2})} {_1F_1}\left(\frac{-\nu}{2}; \frac{1}{2}; \frac{x^2}{2}\right) - \frac{\sqrt{2\pi} x}{\Gamma(\frac{-\nu}{2})} {_1F_1}\left(\frac{1-\nu}{2}; \frac{3}{2}; \frac{x^2}{2}\right)\right]$$

**Alternative Form using Hermite Functions:**
For integer $n$:
$$D_n(x) = 2^{-n/2} e^{-x^2/4} H_n\left(\frac{x}{\sqrt{2}}\right)$$

where $H_n$ are the Hermite polynomials.

### Asymptotic Expansions

For large $|x|$:
$$D_\nu(x) \sim x^\nu e^{-x^2/4} \left[1 - \frac{\nu(\nu-1)}{2x^2} + \frac{\nu(\nu-1)(\nu-2)(\nu-3)}{8x^4} + \cdots\right]$$

**Derivation:** The asymptotic expansion can be derived from the integral representation:
$$D_\nu(x) = e^{-x^2/4} \int_{-\infty}^{\infty} t^\nu e^{-t^2/2 + xt} dt$$

For large $x$, the saddle point occurs at $t = x$, leading to the asymptotic series. □

### Connection Formulas

The parabolic cylinder functions satisfy:
$$D_\nu(x) = e^{i\nu\pi} D_\nu(-x) + \frac{2\sqrt{2\pi}}{\Gamma(-\nu)} e^{-i\nu\pi/2} D_{-\nu-1}(ix)$$

---

## Coulomb Functions

### Coulomb Wave Equation

The Coulomb functions are solutions to the Coulomb wave equation:
$$\frac{d^2y}{dr^2} + \left[k^2 - \frac{2\eta k}{r} - \frac{\ell(\ell+1)}{r^2}\right]y = 0$$

where $\eta$ is the Sommerfeld parameter and $k$ is the wave number.

### Regular and Irregular Solutions

**Regular Coulomb Function $F_\ell(\eta, r)$:**
$$F_\ell(\eta, r) = \frac{2^\ell e^{-\pi\eta/2} |\Gamma(\ell+1+i\eta)|}{(2\ell+1)!} (kr)^{\ell+1} e^{ikr} {_1F_1}(\ell+1-i\eta; 2\ell+2; -2ikr)$$

**Irregular Coulomb Function $G_\ell(\eta, r)$:**
$$G_\ell(\eta, r) = \frac{F_\ell(\eta, r) \tan(\sigma_\ell) - F_{-\ell-1}(\eta, r)}{\tan(\sigma_\ell)}$$

where $\sigma_\ell = \arg[\Gamma(\ell+1+i\eta)]$ is the Coulomb phase shift.

### Asymptotic Behavior

For large $r$:
$$F_\ell(\eta, r) \sim \sin\left(kr - \frac{\ell\pi}{2} + \sigma_\ell - \eta \ln(2kr)\right)$$
$$G_\ell(\eta, r) \sim \cos\left(kr - \frac{\ell\pi}{2} + \sigma_\ell - \eta \ln(2kr)\right)$$

### Wronskian Relations

$$F_\ell G_\ell' - F_\ell' G_\ell = 1$$

**Proof:** This follows from the linear independence of $F_\ell$ and $G_\ell$ and their normalization. The Wronskian is constant and equals unity by construction. □

---

## Kelvin Functions

### Kelvin's Differential Equation

The Kelvin functions are solutions to:
$$x^2 \frac{d^2y}{dx^2} + x\frac{dy}{dx} - (ix^2 + \nu^2)y = 0$$

This is the modified Bessel equation with purely imaginary argument.

### Principal Kelvin Functions

**Real Kelvin Functions:**
- $\text{ber}_\nu(x) = \Re[J_\nu(xe^{3\pi i/4})]$
- $\text{bei}_\nu(x) = \Im[J_\nu(xe^{3\pi i/4})]$
- $\text{ker}_\nu(x) = \Re[K_\nu(xe^{3\pi i/4})]$
- $\text{kei}_\nu(x) = \Im[K_\nu(xe^{3\pi i/4})]$

### Series Representations

For $\nu = 0$:
$$\text{ber}(x) = \sum_{k=0}^{\infty} \frac{(-1)^k}{[(2k)!]^2} \left(\frac{x}{2}\right)^{4k}$$

$$\text{bei}(x) = \sum_{k=0}^{\infty} \frac{(-1)^k}{[(2k+1)!]^2} \left(\frac{x}{2}\right)^{4k+2}$$

### Asymptotic Behavior

For large $x$:
$$\text{ber}(x) + i\text{bei}(x) \sim \frac{e^{x/\sqrt{2}}}{\sqrt{2\pi x}} e^{i(x/\sqrt{2} - \pi/8)}$$

$$\text{ker}(x) + i\text{kei}(x) \sim \sqrt{\frac{\pi}{2x}} e^{-x/\sqrt{2}} e^{-i(x/\sqrt{2} - \pi/8)}$$

### Physical Applications

Kelvin functions appear in heat conduction problems in cylindrical geometries with oscillatory boundary conditions, particularly in skin effect calculations for electromagnetic fields in cylindrical conductors.

---

## Information Theory Functions

### Entropy Functions

**Shannon Entropy:**
$$H(p) = -\sum_{i=1}^n p_i \log p_i$$

**Differential Entropy (for continuous distributions):**
$$h(f) = -\int_{-\infty}^{\infty} f(x) \log f(x) dx$$

### Kullback-Leibler Divergence

**Definition:**
$$D_{KL}(P \parallel Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

**Properties:**
1. $D_{KL}(P \parallel Q) \geq 0$ (non-negativity)
2. $D_{KL}(P \parallel Q) = 0$ if and only if $P = Q$
3. $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$ (asymmetric)

**Proof of Non-negativity (Gibbs' Inequality):**
Using Jensen's inequality for the convex function $-\log x$:
$$-\sum_i P(i) \log \frac{Q(i)}{P(i)} \geq -\log \sum_i P(i) \frac{Q(i)}{P(i)} = -\log \sum_i Q(i) = 0$$

Therefore, $D_{KL}(P \parallel Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)} \geq 0$. □

### Cross-Entropy

$$H(P,Q) = -\sum_i P(i) \log Q(i) = H(P) + D_{KL}(P \parallel Q)$$

This decomposition shows that cross-entropy consists of the entropy of $P$ plus the additional "cost" of using $Q$ instead of $P$.

### Mutual Information

$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} = D_{KL}(P_{X,Y} \parallel P_X \otimes P_Y)$$

### Entropy Rate

For a stationary stochastic process $\{X_n\}$:
$$H_\infty = \lim_{n \to \infty} \frac{1}{n} H(X_1, X_2, \ldots, X_n)$$

---

## Spheroidal Wave Functions

### Spheroidal Wave Equation

The spheroidal wave equation in prolate spheroidal coordinates $(\xi, \eta, \phi)$ is:
$$\frac{d}{d\xi}\left[(1-\xi^2)\frac{dy}{d\xi}\right] + \left[\lambda - c^2\xi^2 - \frac{m^2}{1-\xi^2}\right]y = 0$$

### Angular Spheroidal Functions

**Prolate Angular Functions $S_{mn}^{(1)}(c, \eta)$:**
These are solutions to:
$$\frac{d}{d\eta}\left[(1-\eta^2)\frac{dS}{d\eta}\right] + \left[\lambda_{mn}(c) - c^2\eta^2 - \frac{m^2}{1-\eta^2}\right]S = 0$$

**Oblate Angular Functions $S_{mn}^{(3)}(c, \eta)$:**
Solutions to the oblate version with $c$ replaced by $ic$.

### Radial Spheroidal Functions

**Prolate Radial Functions of the First Kind $R_{mn}^{(1)}(c, \xi)$:**
$$R_{mn}^{(1)}(c, \xi) = \sum_{k=0}^{\infty} d_k^{(mn)} (\xi^2 - 1)^{m/2} \xi^{k+m} {_2F_1}\left(-k, k+2m+1; m+\frac{3}{2}; \frac{1-\xi}{2}\right)$$

**Prolate Radial Functions of the Second Kind $R_{mn}^{(2)}(c, \xi)$:**
Constructed to be linearly independent from $R_{mn}^{(1)}$ and satisfy appropriate boundary conditions.

### Eigenvalue Equation

The characteristic values $\lambda_{mn}(c)$ satisfy a continued fraction equation:
$$\lambda = m^2 + 2\sum_{j=1}^{\infty} \frac{c^{2j}}{2j(2j-1)} \prod_{k=1}^{j-1} \frac{1}{(2k+2m-1)(2k+2m+1) - \lambda}$$

### Applications

Spheroidal wave functions are essential for:
- Electromagnetic scattering by prolate and oblate spheroids
- Quantum mechanics of particles in spheroidal potentials
- Acoustic scattering problems
- Signal processing and antenna theory

---

## Advanced Wright Function Theory

### Wright's General Function

**Definition:**
$$\Phi(\alpha, \beta; z) = \sum_{n=0}^{\infty} \frac{z^n}{n! \Gamma(\alpha n + \beta)}$$

This generalizes both the exponential function ($\alpha = 0$) and Mittag-Leffler functions.

### Mellin Transform Representation

$$\Phi(\alpha, \beta; z) = \frac{1}{2\pi i} \int_{\mathcal{C}} \Gamma(-s) \Gamma(\beta + \alpha s) (-z)^s ds$$

where $\mathcal{C}$ is an appropriate contour in the complex plane.

### Asymptotic Analysis via Saddle Points

For large $|z|$ with $\alpha > 0$, the saddle point analysis of the Mellin transform yields:

**Main Asymptotic Term:**
$$\Phi(\alpha, \beta; z) \sim \frac{1}{\sqrt{2\pi\alpha}} z^{(\beta-1)/(2\alpha)} \exp\left(\frac{1}{\alpha} \left(\frac{z}{\alpha}\right)^{1/\alpha}\right) \left[1 + O(|z|^{-1/(2\alpha)})\right]$$

**Derivation:** The saddle point $s_0$ satisfies:
$$\frac{d}{ds}[\ln\Gamma(\beta + \alpha s) + s\ln(-z)] = 0$$

This gives $\alpha\psi(\beta + \alpha s_0) + \ln(-z) = 0$, where $\psi$ is the digamma function.

For large $|z|$, $s_0 \approx (z/\alpha)^{1/\alpha}/\alpha$, leading to the asymptotic formula. □

### Special Cases and Relationships

**Wright Bessel Function:**
$$J_{\rho,\beta}(z) = \Phi\left(\rho, \beta; -z\right)$$

**Mittag-Leffler Function:**
$$E_{\alpha,\beta}(z) = \Phi(\alpha, \beta; z)$$

**Generalized Exponential:**
$$E_\alpha(z) = \Phi(\alpha, 1; z)$$

### Fractional Calculus Applications

Wright functions appear naturally in solutions to fractional differential equations:

$$\frac{d^\alpha}{dt^\alpha} y(t) = \lambda y(t), \quad y(0) = y_0$$

has the solution:
$$y(t) = y_0 t^{\alpha-1} E_{\alpha,\alpha}(\lambda t^\alpha)$$

where $E_{\alpha,\beta}$ is the two-parameter Mittag-Leffler function.

---

## References

1. Abramowitz, M., & Stegun, I. A. (1972). Handbook of Mathematical Functions. Dover Publications.
2. Olver, F. W. J., et al. (2010). NIST Handbook of Mathematical Functions. Cambridge University Press.
3. Watson, G. N. (1944). A Treatise on the Theory of Bessel Functions. Cambridge University Press.
4. Erdélyi, A., et al. (1953-1955). Higher Transcendental Functions (3 volumes). McGraw-Hill.
5. Andrews, G. E., Askey, R., & Roy, R. (1999). Special Functions. Cambridge University Press.
6. Temme, N. M. (1996). Special Functions: An Introduction to Classical Functions of Mathematical Physics. Wiley.
7. Wright, E. M. (1935). The asymptotic expansion of the generalized Bessel function. Proc. London Math. Soc.
8. Wong, R., & Zhao, Y. Q. (1999). Exponential asymptotics of the Wright Bessel functions. J. Math. Anal. Appl.